import sys
import os
import json
from time import time, sleep
from mini_sdk import (
    enable_debug_mode,
    disable_debug_mode,
    train_info,
    set_progress,
    record_task_log,
    save_task_store_file,
    upload_task_export_file,
    Response,
    FileData,
    FilePath,
    TaskTrainInfo,
)

# Additional imports for SDXL LoRA training
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    DiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig


def main(
    SEA_DATA: str,
    SEA_TASK_ID: str,
    SEA_TASK_LAST_ID: str,
    SEA_TASK_LAST_EXPORT: str,
    SEA_TASK_RETRY: bool,
    param: dict,
):
    # Determine device (CPU or CUDA)
    device_str = str(param.get("device", "cuda")).lower()
    if device_str not in ["cuda", "cpu"]:
        device_str = "cuda"
    use_cuda = device_str == "cuda"
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="fp16") if use_cuda else Accelerator()
    device = accelerator.device

    # Hyperparameters from params or defaults
    train_batch_size = int(param.get("batch_size", 4))
    num_train_epochs = int(param.get("epochs", 1))
    learning_rate = float(param.get("learning_rate", 1e-4))
    # Image resolution (multiple of 32) for training and generation
    resolution = int(param.get("img_size", 512))
    if resolution > 1024:  # cap to 1024 for safety
        resolution = 1024

    # Log the start of training
    record_task_log(
        SEA_TASK_ID,
        f"Starting LoRA training on device={device_str}, dataset path={SEA_DATA}",
    )
    record_task_log(
        SEA_TASK_ID,
        f"Hyperparameters: batch_size={train_batch_size}, epochs={num_train_epochs}, learning_rate={learning_rate}, img_size={resolution}",
    )

    # Load Stable Diffusion XL base model components
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    tokenizer1 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    max_len1 = tokenizer1.model_max_length or 77
    max_len2 = tokenizer2.model_max_length or 77

    # Load models (use float16 on GPU, float32 on CPU)
    dtype = torch.float16 if use_cuda else torch.float32
    text_encoder1 = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
        model_name, subfolder="text_encoder_2", torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32
    )

    # Move models to target device
    text_encoder1.to(device)
    text_encoder2.to(device)
    vae.to(device)
    unet.to(device)
    text_encoder1.eval()
    text_encoder2.eval()
    vae.eval()

    # Freeze all model parameters (train only LoRA adapter params)
    for p in text_encoder1.parameters():
        p.requires_grad = False
    for p in text_encoder2.parameters():
        p.requires_grad = False
    for p in vae.parameters():
        p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = False

    # Set up LoRA and attach to U-Net
    rank = 4
    lora_targets = ["to_q", "to_k", "to_v", "to_out.0"]
    lora_config = LoraConfig(
        r=rank, lora_alpha=rank, target_modules=lora_targets, bias="none"
    )

    # If continuing training, load previous LoRA weights
    if SEA_TASK_LAST_ID:
        prev_lora_path = None
        if os.path.isdir(SEA_TASK_LAST_EXPORT):
            for f in os.listdir(SEA_TASK_LAST_EXPORT):
                if f.endswith(".safetensors"):
                    prev_lora_path = os.path.join(SEA_TASK_LAST_EXPORT, f)
                    break
        if prev_lora_path and os.path.isfile(prev_lora_path):
            unet.add_adapter(lora_config)
            try:
                unet.load_attn_procs(prev_lora_path)
                record_task_log(
                    SEA_TASK_ID,
                    f"Loaded previous LoRA weights from {os.path.basename(prev_lora_path)} for continued training.",
                )
            except Exception as e:
                print(f"Failed to load previous LoRA weights: {e}", file=sys.stderr)
                record_task_log(
                    SEA_TASK_ID, "Previous LoRA not loaded, starting fresh training."
                )
        else:
            unet.add_adapter(lora_config)
    else:
        # Fresh training
        unet.add_adapter(lora_config)
    # Ensure LoRA params use float32 for stability in training
    for _, p in unet.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # Dataset for image-prompt pairs
    class ImgPrompts(Dataset):
        def __init__(self, root_dir, tok1, tok2, res, max_len1, max_len2):
            self.pairs = []
            exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
            for root, _, files in os.walk(root_dir):
                for fn in files:
                    if fn.lower().endswith(exts):
                        img_path = os.path.join(root, fn)
                        txt_path = os.path.splitext(img_path)[0] + ".txt"
                        if os.path.exists(txt_path):
                            self.pairs.append((img_path, txt_path))
            self.tok1 = tok1
            self.tok2 = tok2
            self.res = res
            self.max_len1 = max_len1
            self.max_len2 = max_len2
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        res, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(res),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            img_path, txt_path = self.pairs[idx]
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            ids1 = self.tok1(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_len1,
                return_tensors="pt",
            )["input_ids"][0]
            ids2 = self.tok2(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_len2,
                return_tensors="pt",
            )["input_ids"][0]
            return pixel_values, ids1, ids2

    ds = ImgPrompts(
        SEA_DATA,
        tokenizer1,
        tokenizer2,
        res=resolution,
        max_len1=max_len1,
        max_len2=max_len2,
    )
    num_samples = len(ds)
    if num_samples == 0:
        record_task_log(
            SEA_TASK_ID, "No image-prompt pairs found in the dataset folder.", True
        )
        print("No image-prompt pairs found in the dataset folder.", file=sys.stderr)
        return

    record_task_log(
        SEA_TASK_ID, f"Loaded {num_samples} image-prompt pairs for training."
    )

    # DataLoader for training data
    dl = DataLoader(
        ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: {
            "pixel_values": torch.stack([item[0] for item in batch]),
            "ids1": torch.stack([item[1] for item in batch]),
            "ids2": torch.stack([item[2] for item in batch]),
        },
    )

    # Set up noise scheduler and optimizer
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )

    # Prepare models and data for accelerated training
    unet, optimizer, dl = accelerator.prepare(unet, optimizer, dl)

    steps_per_epoch = max(1, len(dl))
    total_steps = num_train_epochs * steps_per_epoch
    step_count = 0

    # Progress callback using SeaDeep SDK
    def on_progress(percent):
        # Update progress (0.0 to 0.9 range during training)
        set_progress(SEA_TASK_ID, 0.9 * percent / 100.0)

    # Training loop
    unet.train()
    on_progress(1)  # initialize progress at 1%
    for epoch in range(num_train_epochs):
        for batch in dl:
            step_count += 1
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            ids1 = batch["ids1"].to(device)
            ids2 = batch["ids2"].to(device)
            # Encode images to latents with VAE
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * getattr(vae.config, "scaling_factor", 0.18215)
            # Sample random noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # Get text embeddings for conditioning
            with torch.no_grad():
                text_out1 = text_encoder1(ids1)
                text_out2 = text_encoder2(ids2)
                hidden_states = torch.cat(
                    [text_out1.last_hidden_state, text_out2.last_hidden_state], dim=-1
                )
                pooled_embedding = text_out2.text_embeds  # (batch, 1280)
                add_text_ids = (
                    torch.tensor(
                        [resolution, resolution, 0, 0, resolution, resolution],
                        device=device,
                        dtype=text_out1.last_hidden_state.dtype,
                    )
                    .unsqueeze(0)
                    .repeat(latents.shape[0], 1)
                )
            # U-Net forward with added conditioning (LoRA applied internally)
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=hidden_states,
                added_cond_kwargs={
                    "text_embeds": pooled_embedding,
                    "time_ids": add_text_ids,
                },
            ).sample
            # Compute loss against target noise
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            loss = torch.nn.functional.mse_loss(
                noise_pred.float(), target.float(), reduction="mean"
            )
            # Backpropagate
            accelerator.backward(loss)
            # Gradient clipping and optimizer step
            accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            # Update progress percentage
            percent = int(step_count * 100 / total_steps)
            on_progress(min(100, max(1, percent)))
        # Log epoch completion
        record_task_log(SEA_TASK_ID, f"Epoch {epoch+1}/{num_train_epochs} completed.")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Unwrap and move model to CPU for saving
        unet_model = accelerator.unwrap_model(unet)
        unet_model.to("cpu", dtype=torch.float32)
        # Save LoRA adapter weights to safetensors file
        output_filename = "sdxl_lora.safetensors"
        os.makedirs(".", exist_ok=True)
        try:
            unet_model.save_lora_adapter(
                ".", weight_name=output_filename, safe_serialization=True
            )
        except Exception:
            # Fallback for older diffusers versions
            unet_model.save_attn_procs(
                ".", weight_name=output_filename, safe_serialization=True
            )
        record_task_log(
            SEA_TASK_ID, f"LoRA training finished, model saved as {output_filename}"
        )

    # Upload the LoRA weight file as training output
    set_progress(SEA_TASK_ID, 0.95)
    try:
        with open(output_filename, "rb") as f:
            model_bytes = f.read()
    except Exception as e:
        print(f"Error reading output file: {e}", file=sys.stderr)
        record_task_log(SEA_TASK_ID, f"Failed to read output model file: {e}", True)
        return
    result, ok = upload_task_export_file(
        SEA_TASK_ID, [FileData(output_filename, model_bytes)]
    )
    if not ok:
        if result is None:
            print("伺服器異常", file=sys.stderr)
        else:
            print(f"伺服器回傳 {result.msg} (錯誤代碼: {result.code})", file=sys.stderr)
        return

    # Generate a sample image using the trained LoRA (for demonstration)
    prompt = ""
    if num_samples > 0:
        idx = int(time()) % num_samples
        _, txt_path = ds.pairs[idx]
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        except:
            prompt = ""
    if prompt == "":
        prompt = "A photo"
    record_task_log(SEA_TASK_ID, f'Generating sample image with prompt: "{prompt}"')
    # Load base pipeline for generation
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        variant="fp16" if use_cuda else None,
        use_safetensors=True,
    )
    if use_cuda:
        pipe = pipe.to("cuda")
    # Apply LoRA weights to the pipeline's U-Net
    try:
        pipe.unet.load_attn_procs(output_filename)
    except Exception as e:
        print(f"Failed to load LoRA into pipeline for generation: {e}", file=sys.stderr)
    # Generate image (using base model only for simplicity)
    images = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.0,
        num_images_per_prompt=1,
        height=resolution,
        width=resolution,
    ).images
    gen_image = images[0]
    output_image_path = "output_image.png"
    gen_image.save(output_image_path)
    # Upload the generated image to SeaDeep store
    result, ok = save_task_store_file(
        SEA_TASK_ID, [FilePath("output_image.png", output_image_path)]
    )
    if not ok:
        if result is None:
            print("伺服器異常", file=sys.stderr)
        else:
            print(f"伺服器回傳 {result.msg} (錯誤代碼: {result.code})", file=sys.stderr)
        return
    # Log image availability for download
    record_task_log(SEA_TASK_ID, "download:output_image.png")
    set_progress(SEA_TASK_ID, 1.0)
    # Final completion log
    final_msg = f"LoRA training completed, model saved as {output_filename}"
    print(final_msg)
    result, ok = record_task_log(SEA_TASK_ID, final_msg, True)
    if not ok:
        if result is None:
            print("伺服器異常", file=sys.stderr)
        else:
            print(f"伺服器回傳 {result.msg} (錯誤代碼: {result.code})", file=sys.stderr)
        return


if __name__ == "__main__":
    if os.getenv("SEA_TASK_ID"):
        disable_debug_mode()
    else:
        enable_debug_mode()

    SEA_DATA = str(os.getenv("SEA_DATA")) if os.getenv("SEA_DATA") else ""
    SEA_TASK_ID = str(os.getenv("SEA_TASK_ID")) if os.getenv("SEA_TASK_ID") else ""
    SEA_TASK_LAST_ID = (
        str(os.getenv("SEA_TASK_LAST_ID")) if os.getenv("SEA_TASK_LAST_ID") else ""
    )
    SEA_TASK_LAST_EXPORT = (
        str(os.getenv("SEA_TASK_LAST_EXPORT"))
        if os.getenv("SEA_TASK_LAST_EXPORT")
        else ""
    )
    SEA_TASK_RETRY = (
        str(os.getenv("SEA_TASK_RETRY")) if os.getenv("SEA_TASK_RETRY") else "0"
    ) == "1"

    print("資料集路徑 = ", SEA_DATA)
    print("訓練項目編號 = ", SEA_TASK_ID)
    print("繼承訓練項目編號(retrain) = ", SEA_TASK_LAST_ID)
    print("繼承訓練項目產物(retrain) = ", SEA_TASK_LAST_EXPORT)
    print("是否嘗試跑訓練項目 = ", SEA_TASK_RETRY)

    result, ok = train_info(SEA_TASK_ID)
    if not ok:
        if result is None:
            print("伺服器異常", file=sys.stderr)
        else:
            result: Response = result  # type:ignore
            print(f"伺服器回傳 {result.msg} (錯誤代碼: {result.code})", file=sys.stderr)
        exit(1)

    task: TaskTrainInfo = result  # type:ignore
    print("訓練參數 = ", task.param)

    main(
        SEA_DATA,
        SEA_TASK_ID,
        SEA_TASK_LAST_ID,
        SEA_TASK_LAST_EXPORT,
        SEA_TASK_RETRY,
        task.param,  # type:ignore
    )

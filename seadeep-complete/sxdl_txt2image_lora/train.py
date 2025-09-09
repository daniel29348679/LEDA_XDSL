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
    Response,
    FilePath,
    TaskTrainInfo,
    upload_task_export_file,
    FileData,
)
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import base64
import io


def pil_to_b64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def find_lora_path(directory: str):
    # 在 directory 目錄下尋找 .safetensors 檔案
    if not os.path.isdir(directory or ""):
        return None
    safetensors_files = [f for f in os.listdir(directory) if f.endswith(".safetensors")]
    if not safetensors_files:
        return None
    return directory + "/" + safetensors_files[0]  # 回傳找到的第一個檔案完整路徑


def main(
    SEA_DATA: str,
    SEA_TASK_ID: str,
    SEA_TASK_LAST_ID: str,
    SEA_TASK_LAST_EXPORT: str,
    SEA_TASK_RETRY: bool,
    param: dict,
):
    print(f"SEA_DATA: {SEA_DATA}")
    print(f"SEA_TASK_ID: {SEA_TASK_ID}")
    print(f"SEA_TASK_LAST_ID: {SEA_TASK_LAST_ID}")
    print(f"SEA_TASK_LAST_EXPORT: {SEA_TASK_LAST_EXPORT}")
    print(f"SEA_TASK_RETRY: {SEA_TASK_RETRY}")
    device = str(param.get("device", "cuda")).lower()
    if device not in ["cuda", "cpu"]:
        device = "cuda"
    record_task_log(SEA_TASK_ID, f"Using device: {device}")
    set_progress(SEA_TASK_ID, 0.05)

    try:
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
        ).to(device)
        record_task_log(SEA_TASK_ID, "SDXL base model loaded.")
    except Exception as e:
        record_task_log(SEA_TASK_ID, f"Failed to load SDXL base model: {e}", True)
        return
    set_progress(SEA_TASK_ID, 0.15)

    use_refiner = param.get("use_refiner", True)
    if isinstance(use_refiner, str):
        use_refiner = False if use_refiner.lower() in ["false", "0", "no"] else True

    refiner = None
    if use_refiner:
        try:
            refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None,
            ).to(device)
            record_task_log(SEA_TASK_ID, "SDXL refiner model loaded.")
        except Exception as e:
            record_task_log(
                SEA_TASK_ID,
                f"Failed to load SDXL refiner model: {e} (proceeding without refiner)",
            )
            refiner = None
            use_refiner = False
    set_progress(SEA_TASK_ID, 0.25)

    lora_path = find_lora_path(SEA_DATA)
    use_lora = True
    if param.get("use_lora") is not None:
        val = param.get("use_lora")
        if isinstance(val, str):
            use_lora = False if val.lower() in ["false", "0", "no"] else True
        else:
            use_lora = bool(val)

    print(f"LoRA path: {lora_path}, use_lora: {use_lora}")
    lora_dir = os.path.dirname(lora_path)
    lora_name = os.path.basename(lora_path)
    if lora_path and use_lora:
        try:
            base.unet.load_attn_procs(lora_dir, weight_name=lora_name)
            record_task_log(
                SEA_TASK_ID, f"Loaded LoRA weights from {os.path.basename(lora_path)}."
            )
        except Exception as e:
            record_task_log(SEA_TASK_ID, f"Failed to load LoRA weights: {e}")
    else:
        if lora_path and not use_lora:
            record_task_log(
                SEA_TASK_ID, "LoRA weights found but not applied (use_lora=False)."
            )
        else:
            record_task_log(SEA_TASK_ID, "No LoRA weights to apply.")
    set_progress(SEA_TASK_ID, 0.35)

    prompt = (
        param.get("prompt")
        if param.get("prompt") not in [None, ""]
        else "A high quality photo of a cat"
    )
    negative_prompt = (
        param.get("negative_prompt")
        if param.get("negative_prompt") not in [None, ""]
        else None
    )

    try:
        guidance_scale = float(param.get("guidance_scale", 7.0))
    except:
        guidance_scale = 7.0
    try:
        steps = int(param.get("steps", 20))
    except:
        steps = 20
    try:
        num_images = int(param.get("num_images", 1))
    except:
        num_images = 1
    try:
        width = int(param.get("width", 1024))
    except:
        width = 1024
    try:
        height = int(param.get("height", 1024))
    except:
        height = 1024

    record_task_log(SEA_TASK_ID, f"Prompt: {prompt}")
    if negative_prompt:
        record_task_log(SEA_TASK_ID, f"Negative Prompt: {negative_prompt}")
    record_task_log(
        SEA_TASK_ID, f"Generating {num_images} image(s) at {width}x{height}px..."
    )
    set_progress(SEA_TASK_ID, 0.45)

    images = []
    try:
        if use_refiner and refiner is not None:
            high = 0.8
            out_base = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                num_images_per_prompt=num_images,
                guidance_scale=guidance_scale,
                denoising_end=high,
                output_type="latent",
            )
            latents = out_base.images
            out_final = refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=latents,
                num_inference_steps=steps,
                denoising_start=high,
                guidance_scale=1.0,
                num_images_per_prompt=num_images,
            )
            images = out_final.images
        else:
            out_final = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                num_images_per_prompt=num_images,
                guidance_scale=guidance_scale,
            )
            images = out_final.images
    except Exception as e:
        record_task_log(SEA_TASK_ID, f"Image generation failed: {e}", True)
        return

    files_to_upload = []
    uploaded_names = []
    for idx, img in enumerate(images):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        filename = f"result_{idx}.png" if num_images > 1 else "result.png"
        local_path = os.path.join(".", filename)
        try:
            img.save(local_path)
            files_to_upload.append(FilePath(filename, local_path))
            uploaded_names.append(filename)
        except Exception as e:
            record_task_log(SEA_TASK_ID, f"Failed to save image {idx}: {e}")

    if not files_to_upload:
        record_task_log(SEA_TASK_ID, "No images to upload.", True)
        return

    print(f"Uploading {files_to_upload}")
    result, ok = save_task_store_file(SEA_TASK_ID, files_to_upload)
    if not ok:
        if result is None:
            print("伺服器異常", file=sys.stderr)
        else:
            print(f"伺服器回傳 {result.msg} (錯誤代碼: {result.code})", file=sys.stderr)
        return

    filedatas = []
    for f in files_to_upload:
        try:
            with open(f.path, "rb") as fp:
                bytes_data = fp.read()
                filedatas.append(FileData(f.name, bytes_data))
        except Exception as e:
            record_task_log(
                SEA_TASK_ID, f"Failed to read image {f.name} for upload: {e}"
            )
    result, ok = upload_task_export_file(SEA_TASK_ID, filedatas)
    if not ok:
        if result is None:
            print("伺服器異常", file=sys.stderr)
        else:
            print(f"伺服器回傳 {result.msg} (錯誤代碼: {result.code})", file=sys.stderr)
        return

    for name in uploaded_names:
        record_task_log(SEA_TASK_ID, "download:" + name)

    set_progress(SEA_TASK_ID, 1.0)
    final_msg = f"Image generation completed. {len(uploaded_names)} file(s) uploaded."
    # clean up local files
    for f in files_to_upload:
        try:
            os.remove(f.path)
        except:
            pass
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

    result, ok = train_info(SEA_TASK_ID)  # type: ignore
    if not ok:
        if result is None:
            print("伺服器異常", file=sys.stderr)
        else:
            result: Response = result  # type: ignore
            print(f"伺服器回傳 {result.msg} (錯誤代碼: {result.code})", file=sys.stderr)
        exit(1)

    task: TaskTrainInfo = result  # type: ignore
    print("訓練參數 = ", task.param)

    main(
        SEA_DATA,
        SEA_TASK_ID,
        SEA_TASK_LAST_ID,
        SEA_TASK_LAST_EXPORT,
        SEA_TASK_RETRY,
        task.param,  # type: ignore
    )

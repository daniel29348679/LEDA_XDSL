from diffusers import DiffusionPipeline
import torch
from flask import Flask, request, jsonify
import base64, io, os, threading, shutil, time
from collections import defaultdict
from PIL import Image

# ---------------- core config ----------------
session_data_path = "./session_data"
os.makedirs(session_data_path, exist_ok=True)

session_counter = 0
training_status = {}  # {sid: {status, progress, message?, lora?}}

# per-session lock (Windows 檔案鎖容易衝突；用 lock + tmp 檔 + replace)
_session_locks = defaultdict(threading.Lock)


def session_lock(sid: str):
    return _session_locks[str(sid)]


# ---------------- SDXL base+refiner ----------------
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")


# ---------------- utils ----------------
def pil_to_b64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def session_dir(sid: str) -> str:
    return os.path.join(session_data_path, f"session_{sid}")


def ensure_session(sid: str):
    os.makedirs(session_dir(sid), exist_ok=True)


def latest_lora_path(sid: str):
    d = session_dir(sid)
    if not os.path.isdir(d):
        return None
    cands = []
    for f in os.listdir(d):
        if f.startswith("lora_") and f.endswith(".safetensors"):
            try:
                cands.append((int(f[5:-12]), f))
            except:
                pass
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return os.path.join(d, cands[-1][1])


def next_lora_filename(sid: str) -> str:
    d = session_dir(sid)
    nmax = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.startswith("lora_") and f.endswith(".safetensors"):
                try:
                    nmax = max(nmax, int(f[5:-12]))
                except:
                    pass
    return f"lora_{nmax+1}.safetensors"


# ---------------- generate (base+refiner, optional LoRA) ----------------
def generate_sdxl(
    prompt: str,
    num_images=1,
    height=1024,
    width=1024,
    guidance_scale=7.0,
    steps=20,
    use_lora=False,
    sid=None,
    negative_prompt: str | None = None,
):
    # optional LoRA on base
    if use_lora and sid is not None:
        lp = latest_lora_path(str(sid))
        if not lp or not os.path.exists(lp):
            return None, "No LoRA for this session."
        try:
            with session_lock(sid):
                base.unet.load_attn_procs(lp)
        except Exception as e:
            return None, f"Load LoRA failed: {e}"

    high = 0.8  # 80/20 split

    # Base 階段：可用 CFG
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
    latents = out_base.images  # tensor (B=num_images, C, H', W')

    # Refiner 階段：關掉 CFG，並指明 num_images_per_prompt
    out = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=latents,
        num_inference_steps=steps,
        denoising_start=high,
        guidance_scale=1.0,  # 關 CFG（避免再翻倍）
        num_images_per_prompt=num_images,  # 關鍵：跟 latent 批次一致
    )
    return out.images, None


# ---------------- real SDXL LoRA training (inline, with added_cond_kwargs) ----------------
def train_lora_sdxl(session_folder: str, output_path: str, on_progress=None):
    import os, math, torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms
    from accelerate import Accelerator
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from peft import LoraConfig

    # hyperparams（保守：先穩定）
    train_batch_size = 4
    num_train_epochs = 1
    learning_rate = 1e-4
    max_grad_norm = 1.0
    rank = 4
    lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    resolution = 512

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    tokenizer1 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    max_len1 = tokenizer1.model_max_length or 77
    max_len2 = tokenizer2.model_max_length or 77

    text_encoder1 = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.float16
    )
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
        model_name, subfolder="text_encoder_2", torch_dtype=torch.float16
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")  # fp32 穩定

    text_encoder1.to(device).eval()
    text_encoder2.to(device).eval()
    unet.to(device)
    vae.to(device, dtype=torch.float32).eval()

    for p in (
        list(text_encoder1.parameters())
        + list(text_encoder2.parameters())
        + list(vae.parameters())
    ):
        p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=rank, lora_alpha=rank, target_modules=lora_target_modules, bias="none"
    )
    unet.add_adapter(lora_cfg)
    for _, p in unet.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    class ImgPrompts(Dataset):
        def __init__(self, root, tok1, tok2, res=512):
            self.pairs = []
            exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
            for r, _, fs in os.walk(root):
                for fn in fs:
                    if fn.lower().endswith(exts):
                        ip = os.path.join(r, fn)
                        tp = os.path.splitext(ip)[0] + ".txt"
                        if os.path.exists(tp):
                            self.pairs.append((ip, tp))
            self.tok1, self.tok2 = tok1, tok2
            self.tf = transforms.Compose(
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

        def __getitem__(self, i):
            ip, tp = self.pairs[i]
            img = Image.open(ip).convert("RGB")
            px = self.tf(img)
            with open(tp, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            e1 = self.tok1(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=max_len1,
                return_tensors="pt",
            )["input_ids"][0]
            e2 = self.tok2(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=max_len2,
                return_tensors="pt",
            )["input_ids"][0]
            return px, e1, e2

    ds = ImgPrompts(session_folder, tokenizer1, tokenizer2, res=resolution)
    if len(ds) == 0:
        raise RuntimeError("No image+prompt pairs (*.png/jpg + .txt) found.")

    dl = DataLoader(
        ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: {
            "pixel_values": torch.stack([x[0] for x in b]),
            "ids1": torch.stack([x[1] for x in b]),
            "ids2": torch.stack([x[2] for x in b]),
        },
    )

    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    opt = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )

    unet, opt, dl = accelerator.prepare(unet, opt, dl)

    steps_per_epoch = max(1, len(dl))
    total_steps = num_train_epochs * steps_per_epoch
    step_idx = 0
    if on_progress:
        on_progress(1)

    unet.train()
    for _ in range(num_train_epochs):
        for batch in dl:
            step_idx += 1
            px = batch["pixel_values"].to(device, dtype=torch.float32)
            ids1 = batch["ids1"].to(device)
            ids2 = batch["ids2"].to(device)

            with torch.no_grad():
                lat = vae.encode(px).latent_dist.sample()
                lat = lat * getattr(vae.config, "scaling_factor", 0.18215)

            noise = torch.randn_like(lat)
            t = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (lat.shape[0],),
                device=device,
                dtype=torch.long,
            )
            nlat = noise_scheduler.add_noise(lat, noise, t)

            # SDXL: added_cond_kwargs = {text_embeds, time_ids}
            with torch.no_grad():
                out1 = text_encoder1(ids1)
                out2 = text_encoder2(ids2)
                h1 = out1.last_hidden_state
                h2 = out2.last_hidden_state
                ctx = torch.cat([h1, h2], dim=-1)  # (B, 77, 2048)
                pooled = out2.text_embeds  # (B, 1280)
                add_time_ids = (
                    torch.tensor(
                        [resolution, resolution, 0, 0, resolution, resolution],
                        device=device,
                        dtype=h1.dtype,
                    )
                    .unsqueeze(0)
                    .repeat(lat.shape[0], 1)
                )

            pred = unet(
                nlat,
                t,
                encoder_hidden_states=ctx,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
            ).sample

            target = (
                noise
                if noise_scheduler.config.prediction_type == "epsilon"
                else noise_scheduler.get_velocity(lat, noise, t)
            )
            loss = torch.nn.functional.mse_loss(
                pred.float(), target.float(), reduction="mean"
            )
            accelerator.backward(loss)
            if max_grad_norm is not None:
                accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()

            if on_progress:
                pct = int(step_idx * 100 / total_steps)
                on_progress(min(100, max(1, pct)))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_model = accelerator.unwrap_model(unet)
        unet_model.to("cpu", dtype=torch.float32)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # diffusers 新版：save_lora_adapter；舊版 fallback: save_attn_procs
        try:
            # diffusers>=0.39
            unet_model.save_lora_adapter(
                os.path.dirname(output_path),
                weight_name=os.path.basename(output_path),
                safe_serialization=True,
            )
        except Exception:
            # 舊版相容
            unet_model.save_attn_procs(
                os.path.dirname(output_path),
                weight_name=os.path.basename(output_path),
                safe_serialization=True,
            )


# ---------------- training thread (tmp→replace + lock + retry) ----------------
def train_lora_thread(sid: str):
    d = session_dir(sid)
    if not os.path.isdir(d):
        training_status[sid] = {"status": "error", "message": "Session not found"}
        return

    # quick data check
    has_img = False
    for root, _, fs in os.walk(d):
        for fn in fs:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                if os.path.exists(os.path.splitext(os.path.join(root, fn))[0] + ".txt"):
                    has_img = True
                    break
        if has_img:
            break
    if not has_img:
        training_status[sid] = {
            "status": "error",
            "message": "No training images(+txt)",
        }
        return

    training_status[sid] = {"status": "in-progress", "progress": 0}
    tmp_outp = None
    try:
        fname = next_lora_filename(sid)
        outp = os.path.join(d, fname)
        tmp_outp = outp + ".tmp"

        def on_prog(p):
            training_status[sid]["progress"] = int(p)

        # 訓練→寫 tmp
        train_lora_sdxl(d, tmp_outp, on_progress=on_prog)

        # 換名：持鎖，避免同時被讀；遇鎖檔重試
        with session_lock(sid):
            for _ in range(10):
                try:
                    if os.path.exists(outp):
                        os.remove(outp)
                    os.replace(tmp_outp, outp)
                    break
                except PermissionError:
                    time.sleep(0.2)
            else:
                raise PermissionError(f"Could not replace {outp} (file locked)")

        training_status[sid] = {"status": "completed", "progress": 100, "lora": fname}
    except Exception as e:
        try:
            if tmp_outp and os.path.exists(tmp_outp):
                os.remove(tmp_outp)
        except:
            pass
        training_status[sid] = {"status": "error", "message": str(e)}


# ---------------- Flask (JSON-only inputs except create/list) ----------------
app = Flask(__name__)


@app.get("/create_session")
def create_session():
    import uuid

    session_id = str(uuid.uuid4())
    ensure_session(session_id)
    return jsonify({"session_id": session_id})


@app.post("/post_session_image")
def post_session_image():
    data = request.get_json(force=True)
    sid = str(data.get("session_id", "")).strip()
    b64 = data.get("image_base64", "")
    prompt = data.get("prompt", "")
    if not sid or not b64 or not prompt:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "missing session_id or image_base64 or prompt",
                }
            ),
            400,
        )
    ensure_session(sid)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    d = session_dir(sid)
    idx = len([x for x in os.listdir(d)])  # naive
    sub = os.path.join(d, str(idx))
    os.makedirs(sub, exist_ok=True)
    img.save(os.path.join(sub, "0.png"))
    with open(os.path.join(sub, "0.txt"), "a", encoding="utf-8") as f:
        f.write(prompt + "\n")
    return jsonify({"status": "success"})


# 1) start LoRA training
@app.post("/train_lora")
def train_lora_api():
    data = request.get_json(force=True)
    sid = str(data.get("session_id", "")).strip()
    if not sid:
        return jsonify({"status": "error", "message": "Missing session_id"}), 400
    ensure_session(sid)
    training_status[sid] = {"status": "in-progress", "progress": 0}
    threading.Thread(target=train_lora_thread, args=(sid,), daemon=True).start()
    return jsonify({"status": "started"})


# 2) query training status
@app.post("/train_status")
def train_status_api():
    data = request.get_json(force=True)
    sid = str(data.get("session_id", "")).strip()
    if not sid:
        return jsonify({"status": "error", "message": "Missing session_id"}), 400
    st = training_status.get(sid)
    if not st:
        return jsonify({"status": "none", "progress": 0})
    return jsonify(st)


# 3) get latest lora (filename + base64)
@app.post("/get_lora")
def get_lora_api():
    data = request.get_json(force=True)
    sid = str(data.get("session_id", "")).strip()
    iteration = data.get("iteration")
    if not sid:
        return jsonify({"status": "error", "message": "Missing session_id"}), 400
    d = session_dir(sid)
    if not os.path.isdir(d):
        return jsonify({"status": "error", "message": "Session not found"}), 404

    if iteration is not None:
        try:
            it = int(iteration)
        except:
            return jsonify({"status": "error", "message": "Invalid iteration"}), 400
        path = os.path.join(d, f"lora_{it}.safetensors")
    else:
        path = latest_lora_path(sid)

    if not path or not os.path.exists(path):
        return jsonify({"status": "error", "message": "Requested LoRA not found"}), 404

    with session_lock(sid):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    return jsonify(
        {"session_id": sid, "lora_filename": os.path.basename(path), "lora_base64": b64}
    )


# 4) txt2img
@app.post("/txt2img")
def txt2img():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"status": "error", "message": "Missing prompt"}), 400

    num_images = int(data.get("num_images", 1))
    height = int(data.get("height", 1024))
    width = int(data.get("width", 1024))
    guidance_scale = float(data.get("guidance_scale", 7.0))
    steps = int(data.get("steps", 20))
    negative_prompt = data.get("negative_prompt")  # <--- 新增: 可選
    sid = data.get("session_id")
    use_lora = bool(data.get("use_lora", False)) and sid is not None

    images, err = generate_sdxl(
        prompt=prompt,
        num_images=num_images,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        steps=steps,
        use_lora=use_lora,
        sid=sid,
        negative_prompt=negative_prompt,  # <--- 往下送
    )
    if images is None:
        return jsonify({"status": "error", "message": err}), 400

    return jsonify({"images_base64": [pil_to_b64(im) for im in images]})


# upload LoRA（tmp→replace + lock）
@app.post("/upload_lora")
def upload_lora():
    data = request.get_json(force=True)
    sid = str(data.get("session_id", "")).strip()
    lora_b64 = data.get("lora_base64", "")
    filename = data.get("filename")
    if not sid or not lora_b64:
        return (
            jsonify(
                {"status": "error", "message": "Missing session_id or lora_base64"}
            ),
            400,
        )
    ensure_session(sid)
    d = session_dir(sid)
    fname = filename if filename else next_lora_filename(sid)
    if not fname.endswith(".safetensors"):
        fname += ".safetensors"
    path = os.path.join(d, fname)
    tmp = path + ".tmp"

    with session_lock(sid):
        try:
            with open(tmp, "wb") as f:
                f.write(base64.b64decode(lora_b64))
            if os.path.exists(path):
                os.remove(path)
            os.replace(tmp, path)
        except Exception as e:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except:
                pass
            return jsonify({"status": "error", "message": f"Write failed: {e}"}), 500

    return jsonify({"status": "success", "lora_filename": fname})


# list sessions
@app.get("/list_sessions")
def list_sessions():
    out = []
    for name in os.listdir(session_data_path):
        if not name.startswith("session_"):
            continue
        sid = name.split("_", 1)[1]
        d = os.path.join(session_data_path, name)
        if not os.path.isdir(d):
            continue
        lp = latest_lora_path(sid)
        img_groups = 0
        for sub in os.listdir(d):
            sp = os.path.join(d, sub)
            if os.path.isdir(sp):
                if any(
                    fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
                    for fn in os.listdir(sp)
                ):
                    img_groups += 1
        out.append(
            {
                "session_id": sid,
                "latest_lora": os.path.basename(lp) if lp else None,
                "image_groups": img_groups,
                "training": training_status.get(sid, {"status": "none"}),
            }
        )
    out.sort(key=lambda x: x["session_id"])
    return jsonify({"sessions": out})


# delete session
@app.post("/delete_session")
def delete_session():
    data = request.get_json(force=True)
    sid = str(data.get("session_id", "")).strip()
    if not sid:
        return jsonify({"status": "error", "message": "Missing session_id"}), 400
    d = session_dir(sid)
    if not os.path.isdir(d):
        return jsonify({"status": "error", "message": "Session not found"}), 404
    with session_lock(sid):
        try:
            shutil.rmtree(d)
            training_status.pop(sid, None)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Delete failed: {e}"}), 500
    return jsonify({"status": "deleted", "session_id": sid})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

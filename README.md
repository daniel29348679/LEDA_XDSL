# SDXL LoRA Service — API README

極簡、可維護的 SDXL 服務：

* Session 式資料管理
* 背景 Thread 訓練 **LoRA**（輸出 `.safetensors`）
* 生成 API 預設 **Base+Refiner**（Refiner 關 CFG，穩）
* Windows-friendly：**per-session 檔案鎖 + tmp 檔原子換名**



## 目錄

* [需求](#需求)
* [安裝](#安裝)
* [快速開始](#快速開始)
* [資料結構](#資料結構)
* [API 一覽](#api-一覽)

  * [GET /create\_session](#get-create_session)
  * [POST /post\_session\_image](#post-post_session_image)
  * [POST /train\_lora](#post-train_lora)
  * [POST /train\_status](#post-train_status)
  * [POST /get\_lora](#post-get_lora)
  * [POST /upload\_lora](#post-upload_lora)
  * [GET /list\_sessions](#get-list_sessions)
  * [POST /delete\_session](#post-delete_session)
  * [POST /txt2img](#post-txt2img)
* [訓練行為](#訓練行為)
* [生成行為](#生成行為)
* [錯誤與疑難排解](#錯誤與疑難排解)
* [安全與部署](#安全與部署)
* [版本/相容性備註](#版本相容性備註)

---

## 需求

* Python 3.10–3.12
* NVIDIA GPU（建議 ≥ 16 GB VRAM），CUDA 已安裝
* 主要套件：

  * `torch`, `torchvision`（對應你的 CUDA 版本）
  * `diffusers`（建議 ≥ 0.24）
  * `transformers`
  * `accelerate`
  * `safetensors`
  * `peft`
  * `flask`, `Pillow`

> **注意**：訓練腳本需要 `peft`；若不想裝，可改為純 diffusers LoRA 實作，但目前版本使用 PEFT。

---

## 安裝

```bash
# 建議使用虛擬環境
pip install -U flask pillow accelerate diffusers transformers safetensors peft
# 依你的 CUDA 安裝對應 torch：
# 例：CUDA 12.1
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 快速開始

啟動服務：

   ```bash
   python leda_sdxl.py
   # 預設 http://0.0.0.0:8000
   ```


---

## 資料結構

```
session_data/
  session_1/
    0/                     # 你上傳的第 1 組影像/標註
      0.png
      0.txt                # 對應 prompt（純文字）
    1/
      ...
    lora_1.safetensors     # 第 1 次訓練產物
    lora_2.safetensors     # 第 2 次訓練產物
    ...
  session_2/
  ...
```

* **訓練資料**：任意子資料夾下，只要有影像檔（`.png/.jpg/.jpeg/.webp/.bmp`）且同名 `.txt` prompt 即可訓練。
* **輸出**：每次訓練會在 `session_<id>` 根目錄產生 `lora_<n>.safetensors`。寫入使用 `*.tmp` → `os.replace()` 原子換名，並以 **per-session lock** 避免 Windows 鎖檔問題。

---

## API 一覽

### GET `/create_session`

建立新 Session。

**Request**：無
**Response**

```json
{ "session_id": 1 }
```

---

### POST `/post_session_image`

上傳訓練影像與對應 prompt。

**Request**

```json
{
  "session_id": "1",
  "image_base64": "<PNG/JPEG base64>",
  "prompt": "a red apple on a wooden table"
}
```

**Response**

```json
{ "status": "success" }
```

---

### POST `/train_lora`

啟動該 Session 的 LoRA 訓練，背景 Thread 執行。

**Request**

```json
{ "session_id": "1" }
```

**Response**

```json
{ "status": "started" }
```

---

### POST `/train_status`

查詢訓練狀態。

**Request**

```json
{ "session_id": "1" }
```

**Response**

```json
// 可能回覆
{ "status": "in-progress", "progress": 42 }
{ "status": "completed",  "progress": 100, "lora": "lora_3.safetensors" }
{ "status": "error",      "message": "...錯誤訊息..." }
{ "status": "none",       "progress": 0 }   // 尚無狀態
```

---

### POST `/get_lora`

取得最新或指定版本的 LoRA。

**Request（取最新）**

```json
{ "session_id": "1" }
```

**Request（取第 n 次）**

```json
{ "session_id": "1", "iteration": 2 }
```

**Response**

```json
{
  "session_id": "1",
  "lora_filename": "lora_2.safetensors",
  "lora_base64": "<base64 bytes>"
}
```

---

### POST `/upload_lora`

將現有 LoRA 檔上傳至 Session（不經訓練）。

**Request**

```json
{
  "session_id": "1",
  "lora_base64": "<safetensors base64>",
  "filename": "my_lora.safetensors"  // 可省略，預設 auto 遞增 lora_<n>.safetensors
}
```

**Response**

```json
{ "status": "success", "lora_filename": "lora_5.safetensors" }
```

> **寫入保護**：先寫 `*.tmp`，持鎖後 `os.replace()`；避免 Windows 同名檔案使用中。

---

### GET `/list_sessions`

列出所有 Session 概況。

**Request**：無
**Response**

```json
{
  "sessions": [
    {
      "session_id": "1",
      "latest_lora": "lora_3.safetensors",
      "image_groups": 2,
      "training": { "status":"completed", "progress":100, "lora":"lora_3.safetensors" }
    },
    ...
  ]
}
```

---

### POST `/delete_session`

刪除整個 Session（資料夾 + 訓練狀態）。

**Request**

```json
{ "session_id": "1" }
```

**Response**

```json
{ "status": "deleted", "session_id": "1" }
```

---

### POST `/txt2img`

使用 SDXL Base+Refiner 生成影像；可選擇套用 Session 內最新 LoRA（**僅套在 Base**）。

**Request**

```json
{
  "prompt": "a hyperrealistic red apple, studio lighting, 8k",
  "negative_prompt": "",
  "num_images": 2,
  "height": 1024,
  "width": 1024,
  "guidance_scale": 7.0,
  "steps": 20,
  "use_lora": true,
  "session_id": "1"
}
```

**Response**

```json
{
  "images_base64": ["<png b64>", "<png b64>"]
}
```

**行為說明**

* Base 階段啟用 CFG（`guidance_scale` 依你傳入）。
* Refiner 階段 **固定關閉 CFG（`guidance_scale=1.0`）**，並明確指定 `num_images_per_prompt=num_images`，避免 batch 對不齊錯誤。
* 若 `use_lora=true` 會以 Session 內**最新** LoRA 套到 **Base**（Refiner 不套 LoRA）。

---

## 訓練行為

* 使用 **SDXL Base**（`stabilityai/stable-diffusion-xl-base-1.0`）進行 **LoRA（UNet attention 模組）** 訓練。
* 重要實作：

  * **`added_cond_kwargs`**：在訓練時傳入 `{"text_embeds": pooled, "time_ids": ...}`（SDXL 必備）。
  * VAE 固定 `fp32`（穩定），其餘 `fp16`。
  * 預設超參數（保守）：`batch_size=4`, `epochs=1`, `lr=1e-4`, `rank=4`。
  * 僅訓練 UNet 的 LoRA 參數；Text Encoder/ VAE 冷凍。
* 儲存：

  * **先寫 `*.tmp` → `os.replace()` 原子換名**
  * diffusers 新版優先 `save_lora_adapter()`；舊版 fallback `save_attn_procs()`（向下相容）。

> 想提高品質：自行拉高訓練 epoch / 批次、清洗資料與 prompt。

---

## 生成行為

* Base→Refiner **兩段式**：

  * Base：使用你的 `guidance_scale`、`denoising_end=0.8`，輸出 latent。
  * Refiner：以 `denoising_start=0.8` 接手，**固定 `guidance_scale=1.0`**（關掉 CFG，避免 batch/shape 問題）。
* 若 `use_lora=true` + `session_id`，僅對 **Base** 載入該 Session 最新 LoRA。

---

## 錯誤與疑難排解

| 症狀/訊息                                                                  | 可能原因                     | 解法                                                            |
| ---------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------- |
| `No module named 'peft'`                                               | 未安裝 PEFT                 | `pip install peft`                                            |
| `argument of type 'NoneType' is not iterable`（訓練）                      | 漏傳 `added_cond_kwargs`   | 已在訓練 loop 修正：提供 `text_embeds` + `time_ids`                    |
| `PermissionError: [Errno 13] Permission denied: ...lora_X.safetensors` | Windows 檔案被同時讀/寫         | 已加入 per-session lock + `*.tmp`→`os.replace()` + retry         |
| `The size of tensor a (8192) must match ... (4096)`（生成）                | Refiner CFG 造成 batch 對不齊 | 已將 Refiner 固定 `guidance_scale=1.0` 並傳 `num_images_per_prompt` |
| `Requested LoRA not found`                                             | 訓練未完成或 iteration 不存在     | 先查 `/train_status` 是否 `completed`；或改用最新                       |

---

## 安全與部署

* 依需求調整 Flask 併發（或換成 gunicorn/uvicorn + WSGI/ASGI）。

---

## 版本/相容性備註
* **PyTorch / CUDA** 需與顯卡環境匹配。
---



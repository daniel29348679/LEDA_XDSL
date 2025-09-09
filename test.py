# %%
import requests, base64, time, os, json

# ---------- config (edit these) ----------
BASE_URL = "http://127.0.0.1:8000"

# create session?
DO_CREATE_SESSION = True
# or use existing session id:
SESSION_ID = None  # e.g. "1" (string or int). If None and DO_CREATE_SESSION=True, will create one.

# upload one training image + prompt
POST_IMAGE = True
IMAGE_PATH = "./train_data/1/1.jpg"
PROMPT = "a red apple on a wooden table, studio lighting"

# start training and poll
START_TRAIN = True
POLL_MAX_TRIES = 120
POLL_INTERVAL_SEC = 5

# optionally test getting latest lora
GET_LORA = True
SAVE_LORA_AS = "downloaded_lora.safetensors"

# txt2img test
RUN_TXT2IMG = True
TXT2IMG_PROMPT = "a hyperrealistic red apple on a wooden table, 8k, detailed"
TXT2IMG_USE_LORA = True
TXT2IMG_NUM_IMAGES = 2
OUT_IMG_PREFIX = "gen_"

# list sessions
LIST_SESSIONS = True

# optionally upload a local lora (instead of training)
UPLOAD_LOCAL_LORA = True
LOCAL_LORA_PATH = "downloaded_lora.safetensors"

# optionally delete session at the end
DELETE_SESSION = False
# -----------------------------------------


def j(method, path, payload=None):
    url = f"{BASE_URL}{path}"
    if method == "GET":
        r = requests.get(url, timeout=30)
    else:
        r = requests.request(method, url, json=payload or {}, timeout=6000)
    r.raise_for_status()
    # some endpoints return bytes in base64 inside JSON, safe to parse json always
    return r.json()


def file_to_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_b64_png(b64str, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64str))


def main():
    global SESSION_ID

    # 1) create session
    if DO_CREATE_SESSION or not SESSION_ID:
        resp = j("GET", "/create_session")
        SESSION_ID = str(resp["session_id"])
        print("created session:", SESSION_ID)
    else:
        SESSION_ID = str(SESSION_ID)
        print("using session:", SESSION_ID)

    # 2) optionally upload a lora directly (skip training if you want)
    if UPLOAD_LOCAL_LORA:
        b64 = file_to_b64(LOCAL_LORA_PATH)
        resp = j(
            "POST",
            "/upload_lora",
            {
                "session_id": SESSION_ID,
                "lora_base64": b64,
                "filename": None,  # or "lora_custom.safetensors"
            },
        )
        print("upload_lora:", resp)

    # 3) post one training image + prompt
    if POST_IMAGE:
        img_b64 = file_to_b64(IMAGE_PATH)
        resp = j(
            "POST",
            "/post_session_image",
            {
                "session_id": SESSION_ID,
                "image_base64": img_b64,
                "prompt": PROMPT,
            },
        )
        print("post_session_image:", resp)

    # 4) start training
    if START_TRAIN:
        resp = j("POST", "/train_lora", {"session_id": SESSION_ID})
        print("train_lora:", resp)

        # 5) poll status
        tries = 0
        while tries < POLL_MAX_TRIES:
            tries += 1
            resp = j("POST", "/train_status", {"session_id": SESSION_ID})
            print(f"train_status try {tries}:", resp)
            if resp.get("status") in ("completed", "error"):
                break
            time.sleep(POLL_INTERVAL_SEC)

    # 6) get latest lora
    if GET_LORA:
        try:
            resp = j("POST", "/get_lora", {"session_id": SESSION_ID})
            print("get_lora:", {k: resp[k] for k in ("session_id", "lora_filename")})
            with open(SAVE_LORA_AS, "wb") as f:
                f.write(base64.b64decode(resp["lora_base64"]))
            print("lora saved to:", SAVE_LORA_AS)
        except Exception as e:
            print("get_lora failed:", e)

    # 7) txt2img
    if RUN_TXT2IMG:
        payload = {
            "prompt": TXT2IMG_PROMPT,
            "num_images": TXT2IMG_NUM_IMAGES,
            "height": 1024,
            "width": 1024,
            "guidance_scale": 7.0,
            "steps": 20,
            "use_lora": bool(TXT2IMG_USE_LORA),
            "session_id": SESSION_ID if TXT2IMG_USE_LORA else None,
        }
        resp = j("POST", "/txt2img", payload)
        imgs = resp.get("images_base64", [])
        for i, b in enumerate(imgs):
            path = f"{OUT_IMG_PREFIX}{i}.png"
            save_b64_png(b, path)
            print("saved image:", path)

    # 8) list sessions
    if LIST_SESSIONS:
        resp = j("GET", "/list_sessions")
        print("list_sessions:")
        print(json.dumps(resp, indent=2))

    # 9) delete session (optional)
    if DELETE_SESSION:
        resp = j("POST", "/delete_session", {"session_id": SESSION_ID})
        print("delete_session:", resp)


if __name__ == "__main__":
    main()

# %%

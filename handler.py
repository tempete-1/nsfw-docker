"""
RunPod Serverless Handler — minimal version.
Only 3 actions supported: inpaint, voice (Chatterbox TTS), voice_test (F5-TTS).
Everything else (generate/edit/video/kira/pose_lora/Segformer/ReActor/SeedVR2) was removed
on 2026-04-22 — see git history if you need it back.
"""

import runpod
import json
import urllib.request
import urllib.parse
import time
import base64
import uuid
import subprocess
import sys
import os
import glob as globmod

COMFY_HOST = "127.0.0.1:8188"
comfy_process = None


def start_comfyui():
    """Start ComfyUI subprocess and wait until ready."""
    global comfy_process
    print("Starting ComfyUI...")

    comfy_process = subprocess.Popen(
        [
            sys.executable, "/comfyui/main.py",
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--extra-model-paths-config", "/comfyui/extra_model_paths.yaml",
            "--disable-auto-launch",
            "--preview-method", "none",
        ],
        cwd="/comfyui",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    for i in range(180):
        try:
            urllib.request.urlopen(f"http://{COMFY_HOST}/system_stats")
            print(f"ComfyUI ready after {i+1}s")
            return True
        except Exception:
            time.sleep(1)

    if comfy_process.stdout:
        output = comfy_process.stdout.read().decode(errors='replace')
        print(f"ComfyUI stdout:\n{output[-3000:]}")
    print("ERROR: ComfyUI failed to start")
    return False


def check_models():
    """List inpaint models found on volume."""
    print("=== Checking model files ===")
    for vol in ("/workspace/models", "/runpod-volume/models"):
        if os.path.exists(vol):
            print(f"  Volume mount: {vol}")
            for sub in ("diffusion_models", "vae", "text_encoders"):
                p = os.path.join(vol, sub)
                if os.path.exists(p):
                    for f in os.listdir(p):
                        fpath = os.path.join(p, f)
                        if os.path.isfile(fpath):
                            size_mb = os.path.getsize(fpath) / (1024 * 1024)
                            print(f"    {fpath} ({size_mb:.1f}MB)")
            break
    else:
        print("  WARNING: no volume mount found")

    print("=== Workflows ===")
    for f in globmod.glob("/workflows/*.json"):
        print(f"  {f}")


def queue_prompt(workflow: dict) -> str:
    """Submit workflow to ComfyUI API, return prompt_id."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://{COMFY_HOST}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors='replace')
        print(f"ComfyUI prompt error ({e.code}): {error_body[:2000]}")
        raise RuntimeError(f"ComfyUI rejected workflow: {error_body[:500]}")

    if "error" in resp:
        print(f"ComfyUI prompt response error: {resp['error']}")
        raise RuntimeError(f"ComfyUI workflow error: {resp['error']}")

    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id in response: {resp}")
    return prompt_id


def poll_completion(prompt_id: str, timeout: int = 300) -> dict:
    """Poll ComfyUI until workflow completes."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            raw = urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}").read()
            resp = json.loads(raw)
            if prompt_id in resp:
                entry = resp[prompt_id]
                if "status" in entry:
                    status_str = entry["status"].get("status_str", "")
                    print(f"ComfyUI status: {status_str}")
                    if status_str == "error":
                        messages = entry["status"].get("messages", [])
                        raise RuntimeError(f"ComfyUI execution error: {messages}")
                outputs = entry.get("outputs", {})
                print(f"ComfyUI outputs: {list(outputs.keys())}")
                return outputs
        except (TimeoutError, RuntimeError):
            raise
        except Exception as e:
            print(f"Poll attempt error: {e}")
        time.sleep(2)
    raise TimeoutError("ComfyUI generation timed out")


def get_image_base64(filename: str, subfolder: str, img_type: str) -> str:
    """Fetch generated image from ComfyUI and return as base64."""
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": img_type,
    })
    resp = urllib.request.urlopen(f"http://{COMFY_HOST}/view?{params}")
    return base64.b64encode(resp.read()).decode()


def save_base64_image(b64_data: str, prefix: str = "input") -> str:
    """Save base64 image to ComfyUI input dir, return filename."""
    img_bytes = base64.b64decode(b64_data)
    comfy_input = "/comfyui/input"
    os.makedirs(comfy_input, exist_ok=True)
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    with open(os.path.join(comfy_input, fname), "wb") as f:
        f.write(img_bytes)
    return fname


def save_mask_image(b64_data: str) -> str:
    """Save mask as RGBA. White pixels in mask → transparent (alpha=0) → ComfyUI inpaints there."""
    from PIL import Image, ImageOps
    import io

    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    r, g, b, _ = img.split()
    gray = img.convert("L")
    inv_gray = ImageOps.invert(gray)
    result = Image.merge("RGBA", (r, g, b, inv_gray))

    comfy_input = "/comfyui/input"
    os.makedirs(comfy_input, exist_ok=True)
    fname = f"mask_{uuid.uuid4().hex[:8]}.png"
    result.save(os.path.join(comfy_input, fname), "PNG")
    print(f"  Saved mask with alpha channel: {fname} ({img.size})")
    return fname


def load_workflow(name: str) -> dict:
    """Load workflow template JSON."""
    path = f"/workflows/{name}.json"
    if not os.path.exists(path):
        available = globmod.glob("/workflows/*.json")
        raise FileNotFoundError(f"Workflow not found: {path}. Available: {available}")
    with open(path) as f:
        return json.load(f)


def build_inpaint_workflow(job_input: dict) -> dict:
    """Build ComfyUI inpaint workflow from template + job parameters."""
    workflow = load_workflow("inpaint")

    prompt = job_input.get("prompt", "")
    negative = job_input.get(
        "negative",
        "blurry, ugly, deformed, watermark, text, low quality, cartoon",
    )

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        meta_title = str(node.get("_meta", {}).get("title", "")).lower()

        if class_type == "CLIPTextEncode" and "positive" in meta_title:
            node["inputs"]["text"] = prompt
            print(f"  Set positive prompt on node {node_id}")
        elif class_type == "CLIPTextEncode" and "negative" in meta_title:
            node["inputs"]["text"] = negative
            print(f"  Set negative prompt on node {node_id}")

        if class_type == "KSampler":
            seed = job_input.get("seed", -1)
            if seed == -1:
                import random
                seed = random.randint(0, 2**32 - 1)
            node["inputs"]["seed"] = seed
            if "steps" in job_input:
                node["inputs"]["steps"] = job_input["steps"]
            if "cfg" in job_input:
                node["inputs"]["cfg"] = job_input["cfg"]
            if "denoise" in job_input:
                node["inputs"]["denoise"] = job_input["denoise"]
            print(f"  Set seed={seed} on node {node_id}")

        if class_type == "LoadImage":
            if "mask" in meta_title:
                if "mask" in job_input and job_input["mask"]:
                    fname = save_mask_image(job_input["mask"])
                    node["inputs"]["image"] = fname
                    print(f"  Set mask on node {node_id}")
            elif "photo" in job_input and job_input["photo"]:
                fname = save_base64_image(job_input["photo"], "photo")
                node["inputs"]["image"] = fname
                print(f"  Set photo on node {node_id}")

    return workflow


def generate_voice(text: str, exaggeration: float = 0.7, voice_sample_b64: str = None) -> str:
    """Generate voice audio via Chatterbox TTS in isolated venv. Returns base64 OGG Opus."""
    print(f"  Voice: text={len(text)} chars, exaggeration={exaggeration}, clone={voice_sample_b64 is not None}")

    voice_sample_path = None
    if voice_sample_b64:
        voice_sample_path = os.path.join("/comfyui/input", f"voice_ref_{uuid.uuid4().hex[:8]}.wav")
        with open(voice_sample_path, "wb") as f:
            f.write(base64.b64decode(voice_sample_b64))
        print(f"  Voice: saved reference audio: {voice_sample_path}")

    request = json.dumps({
        "text": text,
        "exaggeration": exaggeration,
        "voice_sample_path": voice_sample_path,
    })

    result = subprocess.run(
        ["/opt/chatterbox-venv/bin/python", "/voice_worker.py"],
        input=request,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Always log stderr — helps diagnose silent-output issues
    if result.stderr:
        print(f"  Chatterbox stderr (last 2000 chars):\n{result.stderr[-2000:]}")

    if result.returncode != 0:
        raise RuntimeError(f"Voice generation failed: {result.stderr[-500:]}")

    response = json.loads(result.stdout)
    return wav_to_ogg_b64(response["audio"], "voice")


def generate_voice_f5(text: str, voice_sample_b64: str = None) -> str:
    """Generate voice via F5-TTS. Returns base64 OGG Opus."""
    print(f"  F5-TTS: text={len(text)} chars")

    voice_sample_path = None
    if voice_sample_b64:
        voice_sample_path = os.path.join("/comfyui/input", f"voice_ref_{uuid.uuid4().hex[:8]}.wav")
        with open(voice_sample_path, "wb") as f:
            f.write(base64.b64decode(voice_sample_b64))

    request = json.dumps({
        "text": text,
        "voice_sample_path": voice_sample_path,
    })

    result = subprocess.run(
        ["/opt/f5tts-venv/bin/python", "/f5_voice_worker.py"],
        input=request,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Always log stderr — worker prints [F5TTS] debug markers that help
    # diagnose ref_text/whisper/silence issues even when the call "succeeds"
    if result.stderr:
        print(f"  F5-TTS stderr (last 2000 chars):\n{result.stderr[-2000:]}")

    if result.returncode != 0:
        raise RuntimeError(f"F5-TTS failed: {result.stderr[-500:]}")

    response = json.loads(result.stdout)
    return wav_to_ogg_b64(response["audio"], "f5voice")


def wav_to_ogg_b64(wav_b64: str, prefix: str) -> str:
    """Convert base64 WAV to base64 OGG Opus (Telegram voice format)."""
    wav_bytes = base64.b64decode(wav_b64)
    wav_path = f"/tmp/{prefix}_{uuid.uuid4().hex[:8]}.wav"
    ogg_path = wav_path.replace(".wav", ".ogg")
    with open(wav_path, "wb") as f:
        f.write(wav_bytes)

    conv = subprocess.run(
        ["ffmpeg", "-i", wav_path, "-c:a", "libopus", "-b:a", "64k", "-y", ogg_path],
        capture_output=True, text=True, timeout=30,
    )
    os.remove(wav_path)

    if conv.returncode != 0:
        print(f"  ffmpeg error: {conv.stderr}")
        raise RuntimeError(f"WAV→OGG conversion failed: {conv.stderr[-200:]}")

    with open(ogg_path, "rb") as f:
        ogg_b64 = base64.b64encode(f.read()).decode()
    os.remove(ogg_path)

    print(f"  {prefix}: OGG ready ({len(ogg_b64)} bytes base64)")
    return ogg_b64


def free_comfy_vram():
    """Tell ComfyUI to unload models from VRAM. Required before TTS to avoid OOM
    (FLUX inpaint holds ~12GB; Chatterbox/F5 needs ~3GB; total exceeds 20GB on A4500)."""
    try:
        data = json.dumps({"unload_models": True, "free_memory": True}).encode()
        req = urllib.request.Request(
            f"http://{COMFY_HOST}/free",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        print("  Freed ComfyUI VRAM before TTS")
    except Exception as e:
        print(f"  Warning: could not free ComfyUI VRAM: {e}")


def handler(job):
    """RunPod serverless handler. Routes: voice / voice_test / inpaint."""
    try:
        job_input = job["input"]
        action = job_input.get("action", "")
        mode = job_input.get("mode", "")
        print(f"Job: action={action}, mode={mode}, prompt={job_input.get('prompt', '')[:80]!r}")

        # ── Voice (Chatterbox) ──
        if action == "voice":
            free_comfy_vram()
            text = job_input.get("prompt", "")
            exaggeration = float(job_input.get("exaggeration", 0.7))
            voice_sample = job_input.get("voice_sample")
            audio_b64 = generate_voice(text, exaggeration, voice_sample)
            return {"status": "success", "audio": audio_b64, "images": []}

        # ── Voice Test (F5-TTS) ──
        if action == "voice_test":
            free_comfy_vram()
            text = job_input.get("prompt", "")
            voice_sample = job_input.get("voice_sample")
            audio_b64 = generate_voice_f5(text, voice_sample)
            return {"status": "success", "audio": audio_b64, "images": []}

        # ── Inpaint ──
        if action == "inpaint" or mode == "inpaint":
            count = min(int(job_input.get("count", 1)), 4)
            print(f"Inpaint count: {count}")

            images = []
            for i in range(count):
                import random
                job_input["seed"] = random.randint(0, 2**32 - 1)
                workflow = build_inpaint_workflow(job_input)
                prompt_id = queue_prompt(workflow)
                print(f"Queued inpaint {i+1}/{count}: {prompt_id}")
                outputs = poll_completion(prompt_id)
                for node_output in outputs.values():
                    if "images" in node_output:
                        for img in node_output["images"]:
                            b64 = get_image_base64(
                                img["filename"],
                                img.get("subfolder", ""),
                                img["type"],
                            )
                            images.append(b64)

            print(f"Total inpaint images: {len(images)}")
            return {"status": "success", "images": images, "videos": []}

        return {
            "status": "error",
            "error": f"Unsupported action/mode: action={action!r} mode={mode!r}. Only inpaint, voice, voice_test are supported.",
            "images": [],
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR: {e}\n{tb}")
        return {"status": "error", "error": str(e), "traceback": tb, "images": []}


# ── Cold start ──
print("=== RunPod ComfyUI Handler Starting (minimal: inpaint + voice + voice_test) ===")
check_models()
if not start_comfyui():
    print("FATAL: ComfyUI did not start")

runpod.serverless.start({"handler": handler})

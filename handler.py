"""
RunPod Serverless Handler for ComfyUI
Starts ComfyUI on cold start, accepts generation requests, returns base64 images.
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
import tempfile

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

    # Wait for ComfyUI to be ready (up to 3 min)
    for i in range(180):
        try:
            urllib.request.urlopen(f"http://{COMFY_HOST}/system_stats")
            print(f"ComfyUI ready after {i+1}s")
            return True
        except Exception:
            time.sleep(1)

    print("ERROR: ComfyUI failed to start")
    return False


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
        error_body = e.read().decode()
        print(f"ComfyUI prompt error ({e.code}): {error_body[:1000]}")
        raise RuntimeError(f"ComfyUI rejected workflow: {error_body[:500]}")
    if "error" in resp:
        print(f"ComfyUI prompt response error: {resp['error']}")
        node_errors = resp.get("node_errors", {})
        if node_errors:
            print(f"Node errors: {json.dumps(node_errors, indent=2)[:1000]}")
        raise RuntimeError(f"ComfyUI workflow error: {resp['error']}")
    return resp["prompt_id"]


def poll_completion(prompt_id: str, timeout: int = 300) -> dict:
    """Poll ComfyUI until workflow completes."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = json.loads(
                urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}").read()
            )
            if prompt_id in resp:
                entry = resp[prompt_id]
                # Check for execution errors
                if "status" in entry:
                    status_info = entry["status"]
                    if status_info.get("status_str") == "error":
                        messages = status_info.get("messages", [])
                        print(f"ComfyUI execution error: {messages}")
                        raise RuntimeError(f"ComfyUI error: {messages}")
                print(f"ComfyUI outputs keys: {list(entry.get('outputs', {}).keys())}")
                for nid, nout in entry.get("outputs", {}).items():
                    print(f"  Node {nid}: {list(nout.keys())}")
                return entry.get("outputs", {})
        except (TimeoutError, RuntimeError):
            raise
        except Exception as e:
            print(f"Poll error: {e}")
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
    """Save base64 image to temp file, return path."""
    img_bytes = base64.b64decode(b64_data)
    path = os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex[:8]}.png")
    with open(path, "wb") as f:
        f.write(img_bytes)
    # Also save to ComfyUI input dir so LoadImage can find it
    comfy_input = "/comfyui/input"
    os.makedirs(comfy_input, exist_ok=True)
    fname = os.path.basename(path)
    comfy_path = os.path.join(comfy_input, fname)
    with open(comfy_path, "wb") as f:
        f.write(img_bytes)
    return fname


def load_workflow(action: str) -> dict:
    """Load workflow template JSON."""
    path = f"/workflows/{action}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Workflow not found: {path}")
    with open(path) as f:
        return json.load(f)


def build_workflow(job_input: dict) -> dict:
    """Build ComfyUI workflow from template + job parameters."""
    action = job_input.get("action", "generate")
    mode = job_input.get("mode", "generate")

    # Map mode to workflow file
    workflow_map = {
        "generate": "generate",
        "inpaint": "inpaint",
        "video": "video",
        "edit_easy": "edit_easy",
        "edit_dark": "edit_dark",
    }
    workflow_name = workflow_map.get(mode, action)
    workflow = load_workflow(workflow_name)

    # Inject prompt
    prompt = job_input.get("prompt", "")
    negative = job_input.get("negative", "blurry, ugly, deformed, low quality")

    # Find and set prompt nodes (by title or class)
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")

        # Set positive prompt
        if class_type == "CLIPTextEncode" and "positive" in str(node.get("_meta", {}).get("title", "")).lower():
            node["inputs"]["text"] = prompt
        elif class_type == "CLIPTextEncode" and "negative" in str(node.get("_meta", {}).get("title", "")).lower():
            node["inputs"]["text"] = negative

        # Set seed to random if needed
        if class_type == "KSampler":
            seed = job_input.get("seed", -1)
            if seed == -1:
                import random
                seed = random.randint(0, 2**32 - 1)
            node["inputs"]["seed"] = seed

            # Set other KSampler params if provided
            if "steps" in job_input:
                node["inputs"]["steps"] = job_input["steps"]
            if "cfg" in job_input:
                node["inputs"]["cfg"] = job_input["cfg"]
            if "denoise" in job_input:
                node["inputs"]["denoise"] = job_input["denoise"]

        # Set LoRA strength
        if class_type in ("LoraLoaderModelOnly", "LoraLoader"):
            if "lora_strength" in job_input:
                node["inputs"]["strength_model"] = job_input["lora_strength"]

        # Set resolution
        if class_type == "EmptyLatentImage":
            if "width" in job_input:
                node["inputs"]["width"] = job_input["width"]
            if "height" in job_input:
                node["inputs"]["height"] = job_input["height"]

        # Set input image
        if class_type == "LoadImage":
            if "photo" in job_input and job_input["photo"]:
                fname = save_base64_image(job_input["photo"], "photo")
                node["inputs"]["image"] = fname
            if "face_image" in job_input and job_input["face_image"]:
                meta_title = str(node.get("_meta", {}).get("title", "")).lower()
                if "face" in meta_title or "reference" in meta_title:
                    fname = save_base64_image(job_input["face_image"], "face")
                    node["inputs"]["image"] = fname

    return workflow


def handler(job):
    """RunPod serverless handler."""
    try:
        job_input = job["input"]

        # Build workflow
        workflow = build_workflow(job_input)
        print(f"Workflow nodes: {list(workflow.keys())}")
        for nid, node in workflow.items():
            print(f"  {nid}: {node.get('class_type')} - {node.get('_meta', {}).get('title', '')}")

        # Queue prompt
        prompt_id = queue_prompt(workflow)
        print(f"Queued prompt: {prompt_id}")

        # Poll for completion
        outputs = poll_completion(prompt_id)

        # Extract images
        images = []
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img in node_output["images"]:
                    b64 = get_image_base64(
                        img["filename"],
                        img.get("subfolder", ""),
                        img["type"],
                    )
                    images.append(b64)

        return {"status": "success", "images": images}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"status": "error", "error": str(e), "images": []}


# ── Cold start: launch ComfyUI ──
print("=== RunPod ComfyUI Handler Starting ===")
if not start_comfyui():
    print("FATAL: ComfyUI did not start")

runpod.serverless.start({"handler": handler})

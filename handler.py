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

    # Wait for ComfyUI to be ready (up to 3 min)
    for i in range(180):
        try:
            urllib.request.urlopen(f"http://{COMFY_HOST}/system_stats")
            print(f"ComfyUI ready after {i+1}s")
            return True
        except Exception:
            time.sleep(1)

    # Print ComfyUI output on failure
    if comfy_process.stdout:
        output = comfy_process.stdout.read().decode(errors='replace')
        print(f"ComfyUI stdout:\n{output[-3000:]}")
    print("ERROR: ComfyUI failed to start")
    return False


def check_models():
    """Check what model files exist on the volume."""
    print("=== Checking model files ===")
    vol = "/runpod-volume"
    if os.path.exists(vol):
        for root, dirs, files in os.walk(vol):
            depth = root.replace(vol, '').count(os.sep)
            if depth < 3:
                for f in files:
                    fpath = os.path.join(root, f)
                    size_mb = os.path.getsize(fpath) / (1024*1024)
                    print(f"  {fpath} ({size_mb:.1f}MB)")
    else:
        print(f"  WARNING: {vol} does not exist!")
        # Check alternative paths
        for alt in ["/workspace", "/runpod-volume", "/models"]:
            if os.path.exists(alt):
                print(f"  Found alternative: {alt}")

    # Check workflows
    print("=== Checking workflows ===")
    for f in globmod.glob("/workflows/*.json"):
        print(f"  {f}")

    # Check ComfyUI extra model paths
    yaml_path = "/comfyui/extra_model_paths.yaml"
    if os.path.exists(yaml_path):
        print(f"=== {yaml_path} ===")
        with open(yaml_path) as f:
            print(f.read())


def get_comfy_logs():
    """Get recent ComfyUI stdout."""
    if comfy_process and comfy_process.stdout:
        try:
            # Non-blocking read
            import select
            if hasattr(select, 'select'):
                while select.select([comfy_process.stdout], [], [], 0)[0]:
                    line = comfy_process.stdout.readline()
                    if line:
                        print(f"[ComfyUI] {line.decode(errors='replace').strip()}")
        except Exception:
            pass


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
        node_errors = resp.get("node_errors", {})
        if node_errors:
            print(f"Node errors: {json.dumps(node_errors, indent=2)[:2000]}")
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

                # Check for execution errors
                if "status" in entry:
                    status_info = entry["status"]
                    status_str = status_info.get("status_str", "")
                    print(f"ComfyUI status: {status_str}")
                    if status_str == "error":
                        messages = status_info.get("messages", [])
                        print(f"ComfyUI execution error messages: {messages}")
                        raise RuntimeError(f"ComfyUI execution error: {messages}")

                outputs = entry.get("outputs", {})
                print(f"ComfyUI outputs: {list(outputs.keys())}")
                for nid, nout in outputs.items():
                    print(f"  Node {nid}: keys={list(nout.keys())}")
                    if "images" in nout:
                        print(f"    Images: {len(nout['images'])} items")
                        for img in nout["images"]:
                            print(f"      {img}")

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
        available = globmod.glob("/workflows/*.json")
        raise FileNotFoundError(f"Workflow not found: {path}. Available: {available}")
    with open(path) as f:
        return json.load(f)


def build_workflow(job_input: dict) -> dict:
    """Build ComfyUI workflow from template + job parameters."""
    action = job_input.get("action", "generate")
    mode = job_input.get("mode", "generate")

    workflow_map = {
        "generate": "generate",
        "inpaint": "inpaint",
        "video": "video",
        "edit_easy": "edit_easy",
        "edit_dark": "edit_dark",
        "edit": "edit_easy",
        "dark_beast": "edit_dark",
    }
    workflow_name = workflow_map.get(mode, workflow_map.get(action, "generate"))
    print(f"Loading workflow: {workflow_name} (action={action}, mode={mode})")
    workflow = load_workflow(workflow_name)

    prompt = job_input.get("prompt", "")
    negative = job_input.get("negative", "blurry, ugly, deformed, low quality")

    # Conditionally add kira_lora if prompt mentions "kira"
    use_kira = "kira" in prompt.lower()
    if use_kira:
        kira_lora_path = "/runpod-volume/models/loras/kira_lora.safetensors"
        if os.path.exists(kira_lora_path):
            # Find the last LoRA node and the KSampler to insert kira between them
            last_lora_id = None
            for nid, n in workflow.items():
                if n.get("class_type") in ("LoraLoaderModelOnly", "LoraLoader"):
                    last_lora_id = nid
            sampler_id = None
            for nid, n in workflow.items():
                if n.get("class_type") == "KSampler":
                    sampler_id = nid

            if last_lora_id and sampler_id:
                kira_id = "99"  # use high ID to avoid conflicts
                workflow[kira_id] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": [last_lora_id, 0],
                        "lora_name": "kira_lora.safetensors",
                        "strength_model": job_input.get("lora_strength", 0.95),
                    },
                    "_meta": {"title": "Character LoRA (Kira)"},
                }
                # Point KSampler to kira LoRA output
                workflow[sampler_id]["inputs"]["model"] = [kira_id, 0]
                print(f"  Added kira_lora (node {kira_id})")
        else:
            print(f"  WARNING: kira_lora not found at {kira_lora_path}, skipping")
    else:
        print("  No kira in prompt, skipping kira_lora")

    # Add PuLID face swap if face_photo is provided
    face_photo = job_input.get("face_photo")
    if face_photo and face_photo.strip():
        pulid_model_path = "/runpod-volume/models/pulid/pulid_flux_v0.9.1.safetensors"
        if os.path.exists(pulid_model_path):
            # Save face image
            face_fname = save_base64_image(face_photo, "face")
            print(f"  Face photo saved: {face_fname}")

            # Find the model node that KSampler uses
            sampler_id = None
            sampler_model_ref = None
            for nid, n in workflow.items():
                if n.get("class_type") == "KSampler":
                    sampler_id = nid
                    sampler_model_ref = n["inputs"]["model"]

            if sampler_id and sampler_model_ref:
                # Add PuLID nodes
                workflow["90"] = {
                    "class_type": "PulidFluxModelLoader",
                    "inputs": {
                        "pulid_file": "pulid_flux_v0.9.1.safetensors",
                    },
                    "_meta": {"title": "PuLID Model"},
                }
                workflow["91"] = {
                    "class_type": "PulidFluxInsightFaceLoader",
                    "inputs": {
                        "provider": "CPU",
                    },
                    "_meta": {"title": "PuLID InsightFace"},
                }
                workflow["92"] = {
                    "class_type": "PulidFluxEvaClipLoader",
                    "inputs": {},
                    "_meta": {"title": "PuLID EVA CLIP"},
                }
                workflow["93"] = {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": face_fname,
                    },
                    "_meta": {"title": "Face Reference"},
                }
                workflow["94"] = {
                    "class_type": "ApplyPulidFlux",
                    "inputs": {
                        "model": sampler_model_ref,
                        "pulid_flux": ["90", 0],
                        "eva_clip": ["92", 0],
                        "face_analysis": ["91", 0],
                        "image": ["93", 0],
                        "weight": 0.9,
                        "start_at": 0.0,
                        "end_at": 1.0,
                    },
                    "_meta": {"title": "Apply PuLID"},
                }
                # Point KSampler to PuLID output
                workflow[sampler_id]["inputs"]["model"] = ["94", 0]
                print(f"  Added PuLID face swap nodes (90-94)")
        else:
            print(f"  WARNING: PuLID model not found at {pulid_model_path}")

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        meta_title = str(node.get("_meta", {}).get("title", "")).lower()

        # Set positive prompt
        if class_type == "CLIPTextEncode" and "positive" in meta_title:
            node["inputs"]["text"] = prompt
            print(f"  Set positive prompt on node {node_id}")
        elif class_type == "CLIPTextEncode" and "negative" in meta_title:
            node["inputs"]["text"] = negative
            print(f"  Set negative prompt on node {node_id}")

        # Set seed
        if class_type == "KSampler":
            seed = job_input.get("seed", -1)
            if seed == -1:
                import random
                seed = random.randint(0, 2**32 - 1)
            node["inputs"]["seed"] = seed
            print(f"  Set seed={seed} on node {node_id}")

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
                print(f"  Set photo on node {node_id}")
            if "face_image" in job_input and job_input["face_image"]:
                if "face" in meta_title or "reference" in meta_title:
                    fname = save_base64_image(job_input["face_image"], "face")
                    node["inputs"]["image"] = fname
                    print(f"  Set face_image on node {node_id}")

    return workflow


def handler(job):
    """RunPod serverless handler."""
    try:
        job_input = job["input"]
        print(f"Job input keys: {list(job_input.keys())}")
        print(f"Action: {job_input.get('action')}, Mode: {job_input.get('mode')}")
        print(f"Prompt: {job_input.get('prompt', '')[:100]}")

        # Build workflow
        workflow = build_workflow(job_input)

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
                    print(f"Fetching image: {img}")
                    b64 = get_image_base64(
                        img["filename"],
                        img.get("subfolder", ""),
                        img["type"],
                    )
                    images.append(b64)

        print(f"Total images: {len(images)}")
        if not images:
            print("WARNING: No images in outputs!")
            print(f"Full outputs: {json.dumps({k: list(v.keys()) for k, v in outputs.items()})}")

        return {"status": "success", "images": images}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR: {e}\n{tb}")
        return {"status": "error", "error": str(e), "traceback": tb, "images": []}


# ── Cold start: launch ComfyUI ──
print("=== RunPod ComfyUI Handler Starting ===")
check_models()
if not start_comfyui():
    print("FATAL: ComfyUI did not start")

runpod.serverless.start({"handler": handler})

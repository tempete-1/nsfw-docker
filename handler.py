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

# ── Auto-detect pose LoRA by prompt keywords ──
# Order matters: first match wins. More specific patterns go first.
POSE_LORAS = [
    {
        "keywords": ["pov blowjob", "pov bj", "pov oral", "pov_blowjob"],
        "file": "pov_blowjob.safetensors",
        "strength": 1.0,
        "trigger": "POV_BLOWJOB",
    },
    {
        "keywords": ["deepthroat", "deep throat", "throat fuck"],
        "file": "blowjob_deepthroat.safetensors",
        "strength": 0.9,
        "trigger": "DTSV",
    },
    {
        "keywords": ["eye contact", "looking at viewer", "looking at camera", "eyes"],
        "file": "eye_contact_blowjob.safetensors",
        "strength": 0.7,
        "trigger": "",
    },
    {
        "keywords": ["double penetration from behind", "dp from behind", "dp anal"],
        "file": "dp_from_behind.safetensors",
        "strength": 0.9,
        "trigger": "",
    },
    {
        "keywords": ["double penetration doggystyle", "dp doggystyle", "dp doggy", "double penetration"],
        "file": "dp_doggystyle.safetensors",
        "strength": 0.95,
        "trigger": "DPDS",
    },
    {
        "keywords": ["blowjob", "bj", "oral", "sucking", "fellatio", "licking cock"],
        "file": "blowjobside.safetensors",
        "strength": 0.8,
        "trigger": "BLOWJOBSIDE",
    },
    {
        "keywords": ["cowgirl", "riding", "on top", "reverse cowgirl", "girl on top"],
        "file": "flux_secrets_cowgirl.safetensors",
        "strength": 1.0,
        "trigger": "NUDE WOMAN AND A MAN, ON TOP OF A MAN, COWGIRL",
    },
    {
        "keywords": ["handjob", "hand job", "stroking cock", "jerking off"],
        "file": "handjob.safetensors",
        "strength": 0.8,
        "trigger": "",
    },
    {
        "keywords": ["doggystyle", "doggy", "from behind", "bent over"],
        "file": "cowgirl.safetensors",
        "strength": 0.7,
        "trigger": "",
    },
    {
        "keywords": ["missionary", "lying on back", "on her back"],
        "file": "cowgirl.safetensors",
        "strength": 0.6,
        "trigger": "",
    },
]

LORA_BASE_PATH = "/runpod-volume/models/loras"


def detect_pose_lora(prompt: str) -> dict | None:
    """Find the best matching pose LoRA based on prompt keywords."""
    prompt_lower = prompt.lower()
    for lora in POSE_LORAS:
        for kw in lora["keywords"]:
            if kw in prompt_lower:
                fpath = os.path.join(LORA_BASE_PATH, lora["file"])
                if os.path.exists(fpath):
                    print(f"  Pose LoRA matched: {lora['file']} (keyword='{kw}', strength={lora['strength']})")
                    return lora
                else:
                    print(f"  Pose LoRA matched but file missing: {fpath}")
    return None


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
    comfy_input = "/comfyui/input"
    os.makedirs(comfy_input, exist_ok=True)
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    comfy_path = os.path.join(comfy_input, fname)
    with open(comfy_path, "wb") as f:
        f.write(img_bytes)
    return fname


def save_mask_image(b64_data: str) -> str:
    """Save mask as PNG with alpha channel. White pixels in mask → transparent (alpha=0)."""
    from PIL import Image
    import io

    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Get luminance from RGB — white areas = mask
    r, g, b, a = img.split()
    gray = img.convert("L")

    # Create new image: original RGB but alpha = inverted luminance
    # White in mask (255) → alpha 0 (transparent = masked area)
    # Black in mask (0) → alpha 255 (opaque = keep area)
    from PIL import ImageOps
    inv_gray = ImageOps.invert(gray)

    # Build RGBA with the inverted mask as alpha
    result = Image.merge("RGBA", (r, g, b, inv_gray))

    comfy_input = "/comfyui/input"
    os.makedirs(comfy_input, exist_ok=True)
    fname = f"mask_{uuid.uuid4().hex[:8]}.png"
    comfy_path = os.path.join(comfy_input, fname)
    result.save(comfy_path, "PNG")
    print(f"  Saved mask with alpha channel: {fname} ({img.size})")
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

    # Use Z-Image for generate, Flux for everything else
    use_zimage = (mode == "generate" and action == "generate")

    workflow_map = {
        "generate": "generate_zimage" if use_zimage else "generate",
        "inpaint": "inpaint",
        "video": "video",
        "edit_easy": "edit_easy",
        "edit_dark": "edit_dark",
        "edit": "edit_easy",
        "dark_beast": "edit_dark",
    }
    workflow_name = workflow_map.get(mode, workflow_map.get(action, "generate"))
    print(f"Loading workflow: {workflow_name} (action={action}, mode={mode}, zimage={use_zimage})")
    workflow = load_workflow(workflow_name)

    prompt = job_input.get("prompt", "")
    negative = job_input.get("negative", "blurry, ugly, deformed, low quality, extra fingers, mutated hands, bad hands, malformed limbs, extra limbs, fused fingers, too many fingers, bad anatomy, disfigured")

    # Z-Image: add trigger word, skip pose LoRA (not compatible)
    if use_zimage:
        if "99bsy99" not in prompt:
            prompt = "99bsy99, " + prompt
            job_input["prompt"] = prompt
            print(f"  Added Z-Image trigger: 99bsy99")

    # ── Auto-detect and add pose LoRA (Flux only) ──
    pose_lora = detect_pose_lora(prompt) if not use_zimage else None
    if pose_lora:
        # Add trigger words to prompt if specified
        if pose_lora["trigger"]:
            prompt = pose_lora["trigger"] + ", " + prompt
            job_input["prompt"] = prompt
            print(f"  Prepended trigger: {pose_lora['trigger']}")

        # Find the last LoRA node and KSampler to chain pose LoRA
        last_model_ref = None
        last_lora_id = None
        for nid, n in workflow.items():
            if n.get("class_type") in ("LoraLoaderModelOnly", "LoraLoader"):
                last_lora_id = nid
        sampler_id = None
        for nid, n in workflow.items():
            if n.get("class_type") == "KSampler":
                sampler_id = nid

        if sampler_id:
            # Chain: existing LoRA(s) → pose LoRA → KSampler
            if last_lora_id:
                model_source = [last_lora_id, 0]
            else:
                model_source = workflow[sampler_id]["inputs"]["model"]

            pose_node_id = "80"
            workflow[pose_node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": model_source,
                    "lora_name": pose_lora["file"],
                    "strength_model": pose_lora["strength"],
                },
                "_meta": {"title": f"Pose LoRA ({pose_lora['file']})"},
            }
            workflow[sampler_id]["inputs"]["model"] = [pose_node_id, 0]
            print(f"  Added pose LoRA node {pose_node_id}: {pose_lora['file']}")
    else:
        print("  No pose LoRA matched")

    # ── Conditionally add kira_lora if prompt mentions "kira" ──
    use_kira = "kira" in prompt.lower()
    if use_kira:
        # Z-Image uses ZiT-Lora-loader, Flux uses standard LoraLoaderModelOnly
        if use_zimage:
            kira_lora_path = os.path.join(LORA_BASE_PATH, "kira_lora_zimage.safetensors")
            if os.path.exists(kira_lora_path):
                sampler_id = None
                for nid, n in workflow.items():
                    if n.get("class_type") == "KSampler":
                        sampler_id = nid
                if sampler_id:
                    current_model = workflow[sampler_id]["inputs"]["model"]
                    kira_id = "99"
                    workflow[kira_id] = {
                        "class_type": "ZImageTurboLoraLoader",
                        "inputs": {
                            "model": current_model,
                            "lora_name": "kira_lora_zimage_3000.safetensors",
                            "strength_model": job_input.get("lora_strength", 1.20),
                            "auto_convert_qkv": True,
                        },
                        "_meta": {"title": "Character LoRA Kira (Z-Image)"},
                    }
                    workflow[sampler_id]["inputs"]["model"] = [kira_id, 0]
                    print(f"  Added kira_lora_zimage via ZiT-Lora-loader (node {kira_id})")
            else:
                print(f"  WARNING: kira_lora_zimage not found at {kira_lora_path}")
        else:
            kira_lora_path = os.path.join(LORA_BASE_PATH, "kira_lora.safetensors")
            if os.path.exists(kira_lora_path):
                sampler_id = None
                for nid, n in workflow.items():
                    if n.get("class_type") == "KSampler":
                        sampler_id = nid
                if sampler_id:
                    current_model = workflow[sampler_id]["inputs"]["model"]
                    kira_id = "99"
                    workflow[kira_id] = {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {
                            "model": current_model,
                            "lora_name": "kira_lora.safetensors",
                            "strength_model": job_input.get("lora_strength", 0.70),
                        },
                        "_meta": {"title": "Character LoRA (Kira)"},
                    }
                    workflow[sampler_id]["inputs"]["model"] = [kira_id, 0]
                    print(f"  Added kira_lora for Flux (node {kira_id})")
            else:
                print(f"  WARNING: kira_lora not found at {kira_lora_path}")
    else:
        print("  No kira in prompt, skipping kira_lora")

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        meta_title = str(node.get("_meta", {}).get("title", "")).lower()

        # Set positive prompt (CLIPTextEncode or WanClipTextEncode)
        if class_type in ("CLIPTextEncode", "WanClipTextEncode") and "positive" in meta_title:
            prompt_key = "prompt" if "prompt" in node["inputs"] else "text"
            node["inputs"][prompt_key] = prompt
            print(f"  Set positive prompt on node {node_id}")
        elif class_type in ("CLIPTextEncode", "WanClipTextEncode") and "negative" in meta_title:
            prompt_key = "prompt" if "prompt" in node["inputs"] else "text"
            node["inputs"][prompt_key] = negative
            print(f"  Set negative prompt on node {node_id}")

        # Set seed
        if class_type in ("KSampler", "WanI2VSampler"):
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
        if class_type in ("EmptyLatentImage", "EmptySD3LatentImage"):
            if "width" in job_input:
                node["inputs"]["width"] = job_input["width"]
            if "height" in job_input:
                node["inputs"]["height"] = job_input["height"]

        # Set input image
        if class_type == "LoadImage":
            if "mask" in meta_title:
                # Mask image — save with alpha channel
                if "mask" in job_input and job_input["mask"]:
                    fname = save_mask_image(job_input["mask"])
                    node["inputs"]["image"] = fname
                    print(f"  Set mask on node {node_id}")
            elif "photo" in job_input and job_input["photo"]:
                fname = save_base64_image(job_input["photo"], "photo")
                node["inputs"]["image"] = fname
                print(f"  Set photo on node {node_id}")
            if "face_image" in job_input and job_input["face_image"]:
                if "face" in meta_title or "reference" in meta_title:
                    fname = save_base64_image(job_input["face_image"], "face")
                    node["inputs"]["image"] = fname
                    print(f"  Set face_image on node {node_id}")

    return workflow


def build_face_restore_workflow(generated_image_fname: str, original_face_fname: str) -> dict:
    """Build ReActor face swap workflow: paste original face onto generated image."""
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": generated_image_fname},
            "_meta": {"title": "Generated Image"},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": original_face_fname},
            "_meta": {"title": "Original Face"},
        },
        "3": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "input_image": ["1", 0],
                "source_image": ["2", 0],
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "codeformer-v0.1.0.pth",
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.7,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1,
            },
            "_meta": {"title": "ReActor Face Swap"},
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["3", 0],
                "filename_prefix": "nolimits_facerestore",
            },
            "_meta": {"title": "Save Result"},
        },
    }


def handler(job):
    """RunPod serverless handler."""
    try:
        job_input = job["input"]
        print(f"Job input keys: {list(job_input.keys())}")
        print(f"Action: {job_input.get('action')}, Mode: {job_input.get('mode')}")
        print(f"Prompt: {job_input.get('prompt', '')[:100]}")

        mode = job_input.get("mode", "generate")
        action = job_input.get("action", "generate")

        # Face restore for edit modes (keep original face after edit)
        need_face_restore = (
            mode in ("edit_dark", "edit_easy") or action in ("edit", "dark_beast")
        ) and job_input.get("photo")

        original_face_fname = None
        if need_face_restore:
            original_face_fname = save_base64_image(job_input["photo"], "origface")
            print(f"  Saved original face for restore: {original_face_fname}")

        # Face swap for any mode (user uploads face_photo → swap onto generated image)
        face_photo = job_input.get("face_photo")
        face_swap_fname = None
        if face_photo and face_photo.strip():
            face_swap_fname = save_base64_image(face_photo, "faceswap")
            print(f"  Saved face for ReActor swap: {face_swap_fname}")

        count = int(job_input.get("count", 1))
        count = min(count, 4)  # max 4
        print(f"Count: {count}")

        images = []
        for i in range(count):
            import random
            # Set unique seed for each iteration
            job_input["seed"] = random.randint(0, 2**32 - 1)

            # Build workflow
            workflow = build_workflow(job_input)

            # Queue prompt
            prompt_id = queue_prompt(workflow)
            print(f"Queued prompt {i+1}/{count}: {prompt_id}")

            # Poll for completion
            outputs = poll_completion(prompt_id)

            # Extract images / videos
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img in node_output["images"]:
                        print(f"Fetching image: {img}")
                        b64 = get_image_base64(
                            img["filename"],
                            img.get("subfolder", ""),
                            img["type"],
                        )
                        # Face restore pass for edit modes
                        if need_face_restore and original_face_fname:
                            try:
                                print(f"  Running face restore pass...")
                                gen_fname = save_base64_image(b64, "gen")
                                fr_workflow = build_face_restore_workflow(gen_fname, original_face_fname)
                                fr_prompt_id = queue_prompt(fr_workflow)
                                fr_outputs = poll_completion(fr_prompt_id)
                                # Get face-restored image
                                for fr_nid, fr_nout in fr_outputs.items():
                                    if "images" in fr_nout:
                                        for fr_img in fr_nout["images"]:
                                            b64 = get_image_base64(
                                                fr_img["filename"],
                                                fr_img.get("subfolder", ""),
                                                fr_img["type"],
                                            )
                                            print(f"  Face restore done")
                                            break
                                        break
                            except Exception as e:
                                print(f"  Face restore failed, using original: {e}")

                        # ReActor face swap (if face_photo provided)
                        if face_swap_fname:
                            try:
                                print(f"  Running ReActor face swap...")
                                gen_fname = save_base64_image(b64, "forswap")
                                swap_workflow = build_face_restore_workflow(gen_fname, face_swap_fname)
                                swap_prompt_id = queue_prompt(swap_workflow)
                                swap_outputs = poll_completion(swap_prompt_id)
                                for sw_nid, sw_nout in swap_outputs.items():
                                    if "images" in sw_nout:
                                        for sw_img in sw_nout["images"]:
                                            b64 = get_image_base64(
                                                sw_img["filename"],
                                                sw_img.get("subfolder", ""),
                                                sw_img["type"],
                                            )
                                            print(f"  ReActor face swap done")
                                            break
                                        break
                            except Exception as e:
                                print(f"  ReActor face swap failed, using original: {e}")

                        images.append(b64)
                if "gifs" in node_output:
                    for vid in node_output["gifs"]:
                        print(f"Fetching video: {vid}")
                        b64 = get_image_base64(
                            vid["filename"],
                            vid.get("subfolder", ""),
                            vid["type"],
                        )
                        images.append(b64)

        print(f"Total images: {len(images)}")
        if not images:
            print("WARNING: No images in outputs!")
            print(f"Full outputs: {json.dumps({k: list(v.keys()) for k, v in outputs.items()})}")

        # Check for video files in output directory
        videos = []
        for ext in ("*.mp4", "*.webm", "*.webp"):
            for vf in globmod.glob(f"/comfyui/output/{ext}"):
                if "nolimits_video" in vf:
                    with open(vf, "rb") as f:
                        videos.append(base64.b64encode(f.read()).decode())
                    print(f"Found video output: {vf}")

        return {"status": "success", "images": images, "videos": videos}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR: {e}\n{tb}")
        return {"status": "error", "error": str(e), "traceback": tb, "images": []}


# ── Cold start: setup models and launch ComfyUI ──
print("=== RunPod ComfyUI Handler Starting ===")

# Link ReActor/CodeFormer models from network volume to ComfyUI
def link_model(src_paths, dst_path):
    """Try multiple source paths, symlink first found to dst."""
    if os.path.exists(dst_path):
        print(f"  Already exists: {dst_path}")
        return
    for src in src_paths:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            os.symlink(src, dst_path)
            print(f"  Linked: {src} -> {dst_path}")
            return
    print(f"  WARNING: model not found in any of {src_paths}")

print("=== Linking ReActor models ===")
link_model(
    ["/runpod-volume/models/insightface/inswapper_128.onnx",
     "/workspace/models/insightface/inswapper_128.onnx"],
    "/comfyui/models/insightface/inswapper_128.onnx",
)
link_model(
    ["/runpod-volume/models/facerestore_models/codeformer-v0.1.0.pth",
     "/workspace/models/facerestore_models/codeformer-v0.1.0.pth"],
    "/comfyui/models/facerestore_models/codeformer-v0.1.0.pth",
)
check_models()
if not start_comfyui():
    print("FATAL: ComfyUI did not start")

runpod.serverless.start({"handler": handler})

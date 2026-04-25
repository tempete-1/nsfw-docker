#!/bin/bash
# Run this on a temporary RunPod pod with network volume attached at /runpod-volume
# Downloads all required models

set -e

VOLUME="/runpod-volume/models"

echo "=== Setting up Network Volume ==="

# Create directories
mkdir -p $VOLUME/diffusion_models
mkdir -p $VOLUME/loras
mkdir -p $VOLUME/vae
mkdir -p $VOLUME/text_encoders
mkdir -p $VOLUME/pulid
mkdir -p $VOLUME/controlnet

# Install huggingface CLI
pip install "huggingface_hub[cli]" -q

# Login to HuggingFace (needed for FLUX.1-dev gated model)
echo "Login to HuggingFace first: huggingface-cli login --token YOUR_TOKEN"

# ── Download models ──

echo "Downloading FLUX Unchained..."
# This model is on CivitAI - download manually or use wget with direct link
# wget -O $VOLUME/diffusion_models/fluxUnchainedBySCG_hyfu8StepHybridV10.safetensors "CIVITAI_DIRECT_LINK"
echo "MANUAL: Download fluxUnchainedBySCG from CivitAI and place in $VOLUME/diffusion_models/"

echo "Downloading FLUX Fill Dev (for inpaint)..."
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev flux1-fill-dev.safetensors --local-dir $VOLUME/diffusion_models/

echo "Downloading VAE..."
huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir $VOLUME/vae/

echo "Downloading CLIP L..."
huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir $VOLUME/text_encoders/

echo "Downloading T5 XXL FP16..."
huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir $VOLUME/text_encoders/

echo "Downloading NSFW Unlock LoRA..."
# From CivitAI - download manually
echo "MANUAL: Download aidmaNSFWunlock-FLUX-V0.2.safetensors from CivitAI and place in $VOLUME/loras/"

echo "Downloading PuLID model..."
huggingface-cli download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir $VOLUME/pulid/

# ── Video models (Wan2.1 I2V) ──
mkdir -p $VOLUME/wan

echo "Downloading Wan2.1 I2V 14B 480P model..."
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  diffusion_pytorch_model.safetensors \
  --local-dir $VOLUME/wan/Wan2.1-I2V-14B-480P/

echo "Downloading Wan2.1 VAE..."
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  vae/diffusion_pytorch_model.safetensors \
  --local-dir $VOLUME/wan/Wan2.1-I2V-14B-480P/

echo "Downloading Wan2.1 text encoder..."
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  text_encoder/model.safetensors \
  --local-dir $VOLUME/wan/Wan2.1-I2V-14B-480P/

echo "Downloading Wan2.1 CLIP image encoder..."
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  image_encoder/model.safetensors \
  --local-dir $VOLUME/wan/Wan2.1-I2V-14B-480P/

# ── SAM + GroundingDINO (for edit_nude mode) ──
mkdir -p $VOLUME/sams
mkdir -p $VOLUME/grounding-dino

echo "Downloading SAM ViT-H..."
wget -O $VOLUME/sams/sam_vit_h_4b8939.pth \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

echo "Downloading GroundingDINO SwinT..."
wget -O $VOLUME/grounding-dino/groundingdino_swint_ogc.pth \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

echo "Downloading GroundingDINO config..."
wget -O $VOLUME/grounding-dino/GroundingDINO_SwinT_OGC.py \
  "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
wget -O $VOLUME/grounding-dino/GroundingDINO_SwinT_OGC.cfg.py \
  "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

echo ""
echo "=== Manual steps remaining ==="
echo "1. Upload kira_lora.safetensors to $VOLUME/loras/"
echo "2. Download fluxUnchainedBySCG from CivitAI to $VOLUME/diffusion_models/"
echo "3. Download aidmaNSFWunlock-FLUX-V0.2 from CivitAI to $VOLUME/loras/"
echo ""
echo "After all models are in place, verify:"
echo "ls -la $VOLUME/diffusion_models/"
echo "ls -la $VOLUME/loras/"
echo "ls -la $VOLUME/vae/"
echo "ls -la $VOLUME/text_encoders/"
echo "ls -la $VOLUME/pulid/"
echo "ls -la $VOLUME/wan/"
echo ""
echo "=== Done ==="

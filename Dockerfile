FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps + Python 3.12 (required by SeedVR2)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev git wget \
    build-essential cmake ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m ensurepip --upgrade && pip3 install --no-cache-dir --upgrade pip

# Install ComfyUI (latest — must include PR #12717 fix for Z-Image LoRA)
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
WORKDIR /comfyui
RUN git pull origin master
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# Install custom nodes
RUN cd custom_nodes && \
    git clone https://github.com/cubiq/PuLID_ComfyUI.git && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/mav-rik/facerestore_cf.git && \
    git clone https://codeberg.org/Gourieff/comfyui-reactor-node.git && \
    git clone https://github.com/nunchaku-ai/ComfyUI-nunchaku.git && \
    git clone https://github.com/capitan01R/Comfyui-ZiT-Lora-loader.git && \
    git clone https://github.com/cubiq/ComfyUI_InstantID.git && \
    git clone https://github.com/Jonseed/ComfyUI-Detail-Daemon.git && \
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git && \
    git clone https://github.com/rgthree/rgthree-comfy.git && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    git clone https://github.com/storyicon/comfyui_segment_anything.git && \
    git clone https://github.com/StartHua/Comfyui_segformer_b2_clothes.git

# Install node dependencies
RUN pip3 install --no-cache-dir onnxruntime-gpu 2>/dev/null || pip3 install --no-cache-dir onnxruntime
RUN pip3 install --no-cache-dir insightface facexlib
RUN cd custom_nodes/PuLID_ComfyUI && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/comfyui_controlnet_aux && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/ComfyUI-WanVideoWrapper && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/ComfyUI-VideoHelperSuite && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/ComfyUI-KJNodes && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/facerestore_cf && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/comfyui-reactor-node && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/ComfyUI_InstantID && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/ComfyUI-nunchaku && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN pip3 install --no-cache-dir nunchaku 2>/dev/null || true
RUN pip3 install --no-cache-dir ffmpeg-python
RUN cd custom_nodes/ComfyUI-Detail-Daemon && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/ComfyUI-SeedVR2_VideoUpscaler && pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir gguf_connector 2>/dev/null || true
RUN cd custom_nodes/ComfyUI-GGUF && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/comfyui_segment_anything && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
# Pin transformers for GroundingDINO compatibility
RUN pip3 install --no-cache-dir transformers==4.38.2

# Segformer B2 model for handler.py clothing segmentation (~549MB)
RUN pip3 install --no-cache-dir scipy
RUN mkdir -p /models/segformer_b2_clothes && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mattmdjaga/segformer_b2_clothes', local_dir='/models/segformer_b2_clothes')" || \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mattmdjaga/segformer_b2_clothes', local_dir='/models/segformer_b2_clothes')"

# Verify PuLID nodes exist
RUN ls -la custom_nodes/PuLID_ComfyUI/*.py | head -5

# Chatterbox TTS in isolated venv (needs transformers==5.2.0, conflicts with ComfyUI's 4.38.2)
# 1) CUDA torch first, 2) chatterbox --no-deps (so it doesn't overwrite torch), 3) remaining deps
# 4) Patch t3.py: replace broken lazy import with direct module imports
RUN python3 -m venv /opt/chatterbox-venv && \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir --no-deps chatterbox-tts && \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir transformers==5.2.0 diffusers==0.29.0 safetensors conformer omegaconf && \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir numpy librosa s3tokenizer pykakasi pyloudnorm "spacy-pkuseg>=0.0.27" && \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master" && \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir sentencepiece protobuf accelerate
# Patch chatterbox t3.py: transformers 5.2.0 lazy loader can't resolve top-level imports
RUN T3=/opt/chatterbox-venv/lib/python3.12/site-packages/chatterbox/models/t3/t3.py && \
    sed -i 's/from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model/from transformers.models.llama.modeling_llama import LlamaModel\nfrom transformers.models.llama.configuration_llama import LlamaConfig\nfrom transformers.models.gpt2.modeling_gpt2 import GPT2Model\nfrom transformers.models.gpt2.configuration_gpt2 import GPT2Config/' "$T3" && \
    grep -n "from transformers" "$T3"
# Voice generation runs via subprocess using this venv's python
# Chatterbox model (~3GB) downloads to /models/chatterbox on first run (HF_HOME in voice_worker.py)

# F5-TTS in isolated venv (test alternative to Chatterbox)
RUN python3 -m venv /opt/f5tts-venv && \
    /opt/f5tts-venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/f5tts-venv/bin/pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    /opt/f5tts-venv/bin/pip install --no-cache-dir f5-tts numpy

# RunPod SDK + extras
RUN pip3 install --no-cache-dir runpod
RUN pip3 install --no-cache-dir Pillow

# Copy config and handler
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY handler.py /handler.py
COPY voice_worker.py /voice_worker.py
COPY f5_voice_worker.py /f5_voice_worker.py
# Convert voice sample to WAV at build time (ffmpeg already installed in image)
# Skip first 10s (intro/silence), take 30s of clean voice, mono 22kHz
COPY voice_reference.m4a /tmp/voice_source.m4a
RUN ffmpeg -i /tmp/voice_source.m4a -vn -ss 10 -t 30 -ar 22050 -ac 1 /models/default_female_voice.wav && \
    rm /tmp/voice_source.m4a
COPY workflows/ /workflows/

# Create dirs for ReActor models (will be symlinked from volume at runtime)
RUN mkdir -p /comfyui/input \
    /comfyui/models/insightface \
    /comfyui/models/facerestore_models \
    /comfyui/models/instantid \
    /comfyui/models/controlnet \
    /comfyui/models/sams \
    /comfyui/models/grounding-dino

CMD ["python3", "-u", "/handler.py"]

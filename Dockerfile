FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps + Python 3.12 from deadsnakes PPA
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

# No custom nodes — only inpaint + voice modes use base ComfyUI only
RUN pip3 install --no-cache-dir ffmpeg-python

# Chatterbox TTS in isolated venv (needs transformers==5.2.0, conflicts with ComfyUI's)
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
    /opt/f5tts-venv/bin/pip install --no-cache-dir torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    /opt/f5tts-venv/bin/pip install --no-cache-dir nvidia-cuda-nvrtc-cu12 && \
    /opt/f5tts-venv/bin/pip install --no-cache-dir f5-tts numpy && \
    /opt/f5tts-venv/bin/pip uninstall -y torchcodec || true

# Verify F5's bundled clean reference audio exists — f5_voice_worker.py uses it
# as a known-good reference with a hardcoded transcript instead of
# auto-transcribing our noisy podcast clip via whisper.
RUN ls -la /opt/f5tts-venv/lib/python*/site-packages/f5_tts/infer/examples/basic/ && \
    test -f /opt/f5tts-venv/lib/python*/site-packages/f5_tts/infer/examples/basic/basic_ref_en.wav \
        || (echo "FATAL: F5 bundled ref not found — check f5-tts package version" && exit 1)

# RunPod SDK + extras
RUN pip3 install --no-cache-dir runpod
RUN pip3 install --no-cache-dir Pillow

# Copy config and handler (CACHEBUST forces re-copy on every build to avoid stale cache)
ARG CACHEBUST=1
RUN echo "Cachebust: $CACHEBUST"
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY handler.py /handler.py
COPY voice_worker.py /voice_worker.py
COPY f5_voice_worker.py /f5_voice_worker.py
# Convert voice sample to WAV at build time (ffmpeg already installed in image)
# Skip first 10s (intro/silence), take 30s of clean voice, mono 22kHz
COPY voice_reference.m4a /tmp/voice_source.m4a
RUN mkdir -p /models && \
    ffmpeg -i /tmp/voice_source.m4a -vn -ss 10 -t 30 -ar 22050 -ac 1 /models/default_female_voice.wav && \
    rm /tmp/voice_source.m4a
COPY workflows/ /workflows/

# Create input dir (user-uploaded photos for inpaint)
RUN mkdir -p /comfyui/input

CMD ["python3", "-u", "/handler.py"]

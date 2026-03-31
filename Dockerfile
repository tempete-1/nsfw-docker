FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
WORKDIR /comfyui
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# Install custom nodes
RUN cd custom_nodes && \
    git clone https://github.com/cubiq/PuLID_ComfyUI.git && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git

# Install node dependencies
RUN cd custom_nodes/PuLID_ComfyUI && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true
RUN cd custom_nodes/comfyui_controlnet_aux && pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true

# RunPod SDK + extras
RUN pip3 install --no-cache-dir runpod insightface onnxruntime-gpu

# Copy config and handler
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY handler.py /handler.py
COPY workflows/ /workflows/

# Create input dir for uploaded images
RUN mkdir -p /comfyui/input

CMD ["python3", "-u", "/handler.py"]

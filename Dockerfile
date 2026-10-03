FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --upgrade pip setuptools wheel

RUN python3.9 -m pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    transformers==4.31.0 \
    "tokenizers>=0.12.1,<0.14" \
    sentencepiece==0.1.99 \
    shortuuid \
    accelerate==0.21.0 \
    peft==0.4.0 \
    bitsandbytes==0.41.0 \
    numpy \
    scikit-learn==1.2.2 \
    gradio==3.35.2 \
    gradio_client==0.2.9 \
    requests \
    httpx==0.24.0 \
    uvicorn \
    fastapi \
    einops==0.6.1 \
    einops-exts==0.0.4 \
    timm==0.6.13

WORKDIR /workspace

CMD ["/bin/bash"]

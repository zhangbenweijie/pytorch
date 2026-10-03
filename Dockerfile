# PyTorch official image:
# PyTorch 2.0.1 + CUDA 11.7 + cuDNN 8
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# IMPORTANT:
# torch==2.0.1 is already provided by the official base image.
# Install the VCD versions of the remaining packages.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
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

# Verify the key versions during image build.
# If they are wrong, the Docker build itself fails.
RUN python -c "import torch, torchvision, transformers; \
assert torch.__version__.startswith('2.0.1'); \
assert torchvision.__version__.startswith('0.15.2'); \
assert transformers.__version__ == '4.31.0'; \
print('torch =', torch.__version__); \
print('torchvision =', torchvision.__version__); \
print('transformers =', transformers.__version__); \
print('CUDA build =', torch.version.cuda)"

WORKDIR /workspace

CMD ["/bin/bash"]

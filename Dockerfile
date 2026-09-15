# syntax=docker/dockerfile:1

# ============================================================
# LLaVA-v1.5 Research Environment
#
# Linux x86_64 / Python 3.10 / PyTorch 2.1.2 / CUDA 11.8
# cuDNN 8
#
# Environment only:
# - No LLaVA source code
# - No model weights
# - No datasets
# - No personal/cloud paths
# ============================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ------------------------------------------------------------
# 1. Basic environment
# ------------------------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=2

ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}

# Preserve LD_LIBRARY_PATH provided by the NVIDIA base image.

# ------------------------------------------------------------
# 2. System tools and isolated Python 3.10 environment
# ------------------------------------------------------------

RUN test "$(uname -m)" = "x86_64" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        htop \
        libaio-dev \
        libgl1 \
        libglib2.0-0 \
        nano \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        tmux \
        unzip \
        vim \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv \
    && git lfs install --system

# ------------------------------------------------------------
# 3. Python build tools
# ------------------------------------------------------------

RUN python -m pip install \
    pip==24.0 \
    setuptools==69.0.3 \
    wheel==0.42.0 \
    packaging==23.2

# ------------------------------------------------------------
# 4. Embedded version constraints
#
# No additional local file is required.
# These constraints also apply to later pip installs.
# ------------------------------------------------------------

COPY <<'EOF' /opt/llava-constraints.txt
torch==2.1.2+cu118
torchvision==0.16.2+cu118
transformers==4.37.2
tokenizers==0.15.1
sentencepiece==0.1.99
shortuuid==1.0.11
accelerate==0.21.0
peft==0.7.1
bitsandbytes==0.43.1
huggingface-hub==0.20.3
safetensors==0.4.2
numpy==1.26.4
scipy==1.11.4
scikit-learn==1.2.2
pandas==2.1.4
matplotlib==3.8.2
pillow==10.2.0
opencv-python==4.9.0.80
pydantic==2.6.1
fastapi==0.109.2
starlette==0.36.3
gradio==4.16.0
gradio-client==0.8.1
httpx==0.24.0
uvicorn==0.27.1
python-multipart==0.0.9
typer==0.9.0
click==8.1.7
markdown2==2.4.13
requests==2.31.0
einops==0.6.1
einops-exts==0.0.4
timm==0.6.13
deepspeed==0.12.6
ninja==1.11.1.1
psutil==5.9.8
pynvml==11.5.0
wandb==0.16.3
protobuf==4.25.3
datasets==2.16.1
pyarrow==14.0.2
pyarrow-hotfix==0.6
fsspec==2023.10.0
pycocotools==2.0.7
nltk==3.8.1
openpyxl==3.1.2
tensorboard==2.15.2
tqdm==4.66.2
EOF

ENV PIP_CONSTRAINT=/opt/llava-constraints.txt

# ------------------------------------------------------------
# 5. Install CUDA 11.8 PyTorch explicitly
# ------------------------------------------------------------

RUN python -m pip install \
        numpy==1.26.4 \
        ninja==1.11.1.1 \
    && python -m pip install \
        torch==2.1.2+cu118 \
        torchvision==0.16.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------
# 6. Install core, training and evaluation dependencies
#
# Resolve dependencies together under the constraints.
# DeepSpeed CUDA ops will be compiled on demand at runtime.
# ------------------------------------------------------------

RUN DS_BUILD_OPS=0 python -m pip install \
    --no-build-isolation \
    -r /opt/llava-constraints.txt \
    "markdown2[all]"

# ------------------------------------------------------------
# 7. Verify versions AFTER installing dependencies
# ------------------------------------------------------------

RUN python - <<'PY'
import re
import subprocess
import sys

import torch

nvcc = subprocess.check_output(
    ["/usr/local/cuda/bin/nvcc", "--version"],
    text=True,
)

print(nvcc)
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__ == "2.1.2+cu118", torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert re.search(r"release 11\.8\b", nvcc), nvcc

# The official wheel below uses the old C++ ABI.
assert not torch._C._GLIBCXX_USE_CXX11_ABI, \
    "FlashAttention wheel ABI mismatch"
PY

# ------------------------------------------------------------
# 8. FlashAttention official prebuilt wheel
#
# Matches:
# - Linux x86_64
# - CPython 3.10
# - PyTorch 2.1
# - CUDA 11.8
# - CXX11 ABI = FALSE
#
# --no-deps prevents this step from replacing PyTorch.
# Required dependencies were installed above.
# ------------------------------------------------------------

RUN python -m pip install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# ------------------------------------------------------------
# 9. Validate dependency consistency
# ------------------------------------------------------------

RUN python -m pip check

# ------------------------------------------------------------
# 10. Validate imports and final versions
#
# No GPU is required for these build-time checks.
# GPU execution must be tested on the target machine.
# ------------------------------------------------------------

RUN python - <<'PY'
import importlib

import torch
import flash_attn
import flash_attn_2_cuda

assert torch.__version__ == "2.1.2+cu118", torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert flash_attn.__version__ == "2.5.5", flash_attn.__version__

modules = (
    "torchvision",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "deepspeed",
    "gradio",
    "datasets",
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "cv2",
    "wandb",
    "tensorboard",
)

for name in modules:
    importlib.import_module(name)
    print("IMPORT OK:", name)

print("Environment checks passed")
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("flash_attn:", flash_attn.__version__)
PY

CMD ["/bin/bash"]

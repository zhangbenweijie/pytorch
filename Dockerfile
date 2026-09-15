# ============================================================
# LLaVA-v1.5 Research Environment
# Python 3.10 / PyTorch 2.1.2 / CUDA 11.8 / cuDNN 8
# Environment only: no source code, weights, or datasets.
# ============================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ------------------------------------------------------------
# Basic environment
# ------------------------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    MAX_JOBS=2

# ------------------------------------------------------------
# System tools and Python 3.10
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    htop \
    libgl1 \
    libglib2.0-0 \
    nano \
    python3 \
    python3-dev \
    python3-pip \
    python-is-python3 \
    tmux \
    unzip \
    vim \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# ------------------------------------------------------------
# Python packaging tools
# ------------------------------------------------------------

RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    packaging

# ------------------------------------------------------------
# Install CUDA 11.8 PyTorch explicitly
# ------------------------------------------------------------

RUN python -m pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------
# Verify CUDA toolkit and PyTorch CUDA version
# ------------------------------------------------------------

RUN nvcc --version \
    && python -c "import torch; \
assert torch.__version__.split('+')[0] == '2.1.2', torch.__version__; \
assert torch.version.cuda == '11.8', torch.version.cuda; \
print(f'torch={torch.__version__}, torch_cuda={torch.version.cuda}')"

# ------------------------------------------------------------
# LLaVA core dependencies
# ------------------------------------------------------------

RUN python -m pip install --no-cache-dir \
    transformers==4.37.2 \
    tokenizers==0.15.1 \
    sentencepiece==0.1.99 \
    shortuuid \
    accelerate==0.21.0 \
    peft \
    bitsandbytes \
    pydantic \
    "markdown2[all]" \
    numpy \
    scikit-learn==1.2.2 \
    gradio==4.16.0 \
    gradio_client==0.8.1 \
    requests \
    httpx==0.24.0 \
    uvicorn \
    fastapi \
    einops==0.6.1 \
    einops-exts==0.0.4 \
    timm==0.6.13

# ------------------------------------------------------------
# LLaVA training dependencies
# ------------------------------------------------------------

RUN python -m pip install --no-cache-dir \
    deepspeed==0.12.6 \
    ninja \
    wandb

# ------------------------------------------------------------
# FlashAttention
# ------------------------------------------------------------

RUN python -m pip install --no-cache-dir \
    flash-attn==2.5.5 \
    --no-build-isolation

# ------------------------------------------------------------
# CV, scientific computing, evaluation, and experiment tools
# ------------------------------------------------------------

RUN python -m pip install --no-cache-dir \
    datasets \
    matplotlib \
    nltk \
    opencv-python \
    openpyxl \
    pandas \
    pillow \
    pycocotools \
    scipy \
    tensorboard \
    tqdm

# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

RUN python -c "import torch, flash_attn; \
print(f'torch={torch.__version__}'); \
print(f'torch_cuda={torch.version.cuda}'); \
print(f'flash_attn={flash_attn.__version__}')" \
    && pip check \
    && rm -rf /root/.cache/pip

CMD ["/bin/bash"]

# ============================================================
# LLaVA-v1.5 Research Environment
# Python 3.10 / PyTorch 2.1.2 / CUDA 11.8 / cuDNN 8
# Environment only: no source code, weights, or datasets.
# ============================================================

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

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
# System tools
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    git-lfs \
    htop \
    libgl1 \
    libglib2.0-0 \
    nano \
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
# Verify CUDA toolkit and PyTorch CUDA version.
# Both must be CUDA 11.8 before FlashAttention is compiled.
# ------------------------------------------------------------

RUN nvcc --version \
    && python -c "import torch; \
assert torch.__version__ == '2.1.2', torch.__version__; \
assert torch.version.cuda == '11.8', torch.version.cuda; \
print(f'torch={torch.__version__}, torch_cuda={torch.version.cuda}')"

# ------------------------------------------------------------
# LLaVA core dependencies
# Matches LLaVA pyproject.toml.
# ------------------------------------------------------------

RUN pip install --no-cache-dir \
    torchvision==0.16.2 \
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
# Training dependencies
# ------------------------------------------------------------

RUN pip install --no-cache-dir \
    deepspeed==0.12.6 \
    ninja \
    wandb

# ------------------------------------------------------------
# FlashAttention
# Pin version compatible with PyTorch 2.1.x / CUDA 11.8.
# ------------------------------------------------------------

RUN pip install --no-cache-dir \
    flash-attn==2.5.5 \
    --no-build-isolation

# ------------------------------------------------------------
# CV, scientific computing, evaluation, and experiment tools
# ------------------------------------------------------------

RUN pip install --no-cache-dir \
    opencv-python \
    pillow \
    scipy \
    pandas \
    matplotlib \
    tqdm \
    datasets \
    pycocotools \
    nltk \
    openpyxl \
    tensorboard

# ------------------------------------------------------------
# Final dependency validation
# ------------------------------------------------------------

RUN python -c "import torch, flash_attn; \
print(f'torch={torch.__version__}'); \
print(f'torch_cuda={torch.version.cuda}'); \
print(f'flash_attn={flash_attn.__version__}')" \
    && pip check \
    && rm -rf /root/.cache/pip

CMD ["/bin/bash"]

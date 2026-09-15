# ============================================================
# MLLM / LLaVA Research Environment
#
# Environment only:
#   - NO LLaVA source code
#   - NO model weights
#   - NO datasets
#   - NO personal/cloud paths
#
# Target:
#   LLaVA-v1.5
#   Inference / Evaluation
#   Stage-1 Pretraining (Feature Alignment)
#   Stage-2 Visual Instruction Tuning
#   LoRA Fine-tuning
#
# Python : 3.10
# PyTorch: 2.1.2
# CUDA   : 11.8
# cuDNN  : 8
# ============================================================


# ------------------------------------------------------------
# 1. Base image
# ------------------------------------------------------------
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel


# ------------------------------------------------------------
# 2. Basic environment variables
# ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1


# ------------------------------------------------------------
# 3. Linux tools
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    nano \
    tmux \
    htop \
    zip \
    unzip \
    ffmpeg \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# 4. Git LFS
# ------------------------------------------------------------
RUN git lfs install


# ------------------------------------------------------------
# 5. Python packaging tools
# ------------------------------------------------------------
RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    packaging


# ------------------------------------------------------------
# 6. LLaVA core dependencies
#
# Based on official LLaVA pyproject.toml.
# torch is already provided by the base image.
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
# 7. LLaVA training dependencies
#
# Official [train] dependencies:
#   deepspeed
#   ninja
#   wandb
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    deepspeed==0.12.6 \
    ninja \
    wandb


# ------------------------------------------------------------
# 8. FlashAttention
#
# Required/recommended by official LLaVA training setup.
# The "devel" base image provides nvcc for CUDA compilation.
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    flash-attn \
    --no-build-isolation


# ------------------------------------------------------------
# 9. Common CV / scientific computing tools
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    opencv-python \
    pillow \
    scipy \
    pandas \
    matplotlib \
    tqdm


# ------------------------------------------------------------
# 10. Dataset / evaluation tools
#
# Useful for COCO / hallucination evaluation / data processing.
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    datasets \
    pycocotools \
    nltk \
    openpyxl


# ------------------------------------------------------------
# 11. Experiment monitoring
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    tensorboard


# ------------------------------------------------------------
# 12. Clean pip cache
# ------------------------------------------------------------
RUN rm -rf /root/.cache/pip


# ------------------------------------------------------------
# 13. Default command
#
# No project-specific WORKDIR is specified.
# ------------------------------------------------------------
CMD ["/bin/bash"]

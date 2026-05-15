# =========================
# Build the image
# =========================
# nvidia-docker build --target base -f dockers/Dockerfile.gb --tag gitlab-master.nvidia.com:5005/arch_moe_exploration/megatron-moe-scripts:mcore-moe-pytorch25.09-te2.9.0.dev0-cudnn9.14 --network host .

FROM nvcr.io/nvidia/pytorch:26.04-py3 AS base

ENV SHELL=/bin/bash

# =========================
# Install system packages
# =========================
RUN rm -rf /opt/megatron-lm && \
    apt-get update && \
    apt-get install -y sudo gdb bash-builtins git zsh autojump tmux curl gettext libfabric-dev && \
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_arm64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# =========================
# Install Python packages
# =========================
# NOTE: `unset PIP_CONSTRAINT` to install packages that do not meet the default constraint in the base image.
# Some package requirements and related versions are from 
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/Dockerfile.linting.
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/requirements_mlm.txt.
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.0/requirements_ci.txt.
RUN unset PIP_CONSTRAINT && pip install --no-cache-dir debugpy dm-tree torch_tb_profiler einops wandb \
    sentencepiece tokenizers transformers==4.57.1 torchvision ftfy modelcards datasets tqdm pydantic \
    nvidia-pytriton py-spy yapf darker \
    tiktoken flask-restful \
    nltk wrapt pytest pytest_asyncio pytest-cov pytest_mock pytest-random-order \
    black==24.4.2 isort==5.13.2 flake8==7.1.0 pylint==3.2.6 coverage mypy \
    one-logger --index-url https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple \
    setuptools==69.5.1 nvidia-cutlass-dsl==4.4.2

# =========================
# Install cudnn 9.14.0.64 for correct mxfp8 quantization and layernorm fusion
# =========================
# The latest build is in https://urm.nvidia.com/artifactory/hw-cudnn-generic/CUDNN/
# NOTE: Download (wget --user [username] --ask-password [url]) and extract (tar -xvf *.tar.gz) the package to ./cudnn/cudnn[version]
# If not using the internal build, install it using apt-get
# RUN apt-get update && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb && \
#     dpkg -i cuda-keyring_1.1-1_all.deb && \
#     apt-get update && \
#     apt-get -y install libcudnn9-cuda-13

# The latest build is in https://urm.nvidia.com/artifactory/hw-cudnn-generic/CUDNN/v9.18/v9.18.1.1/13.1/cudnn-linux-sbsa-9.18.1.1.tar.gz
# NOTE: Download (wget --user [username] --ask-password [url]) and extract (tar -xvf *.tar.gz) the package to ./cudnn/cudnn[version]
# COPY ./cudnn/include/ /usr/include/
# COPY ./cudnn/include/ /usr/include/aarch64-linux-gnu/
# COPY ./cudnn/lib/ /usr/lib/aarch64-linux-gnu

# =========================
# Update cublas for better performance
# =========================
# CUDA in the base container is enough.

# =========================
# Install latest TE
# Use a specific commit instead of main to make it more stable.
# This is based on release_v2.9 branch and contains some CPU and quantization optimizations.
# =========================
ARG COMMIT="01aef4fc721bd12fd09cd56d53a314aee1b953d6"
ARG TE="git+https://github.com/NVIDIA/TransformerEngine.git@${COMMIT}"
# ARG COMMIT="b76ba0153912dc2113fa800e956be3c44de9c3a3"
# ARG COMMIT="7dd3914726abb79bc99ff5a5db1449458ed64151"
# ARG TE="git+https://github.com/hxbai/TransformerEngine.git@${COMMIT}"
RUN pip install --no-cache-dir flash-attn-4==4.0.0b4 nvidia-mathdx==25.1.1 && \
    unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="100a;103a" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch pip install --no-build-isolation --no-cache-dir $TE

# =========================
# Install HybridEP
# =========================
WORKDIR /home/
RUN git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && git checkout 1b8f467965bb818bf2f6511e06993f5607e1721f && \
    TORCH_CUDA_ARCH_LIST="10.0" pip install --no-build-isolation .
# tongliu_dev
# RUN git clone --branch hybrid-ep https://github.com/Autumn1998/DeepEP.git && \
#     cd DeepEP && git checkout 3d4a884b8bc4e63786b6d7012f7767fe3c91cfb5 && \
#     TORCH_CUDA_ARCH_LIST="10.0" pip install --no-build-isolation .

# =========================
# Install Fast Hadamard Transform
# =========================
WORKDIR /home/
RUN git clone https://github.com/Dao-AILab/fast-hadamard-transform.git fast-hadamard-transform && \
    cd fast-hadamard-transform && \
    pip install --no-build-isolation .

# =========================
# Install Emerging-Optimizers
# =========================
WORKDIR /home/
RUN git clone https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git && \
    cd Emerging-Optimizers && \
    pip install . --no-build-isolation .

# =========================
# Install dsa_kernels
# =========================
WORKDIR /opt/
RUN git clone --branch nv_dev https://github.com/deepseek-ai/FlashMLA.git && \
    cd FlashMLA && \
    FLASH_MLA_DISABLE_SM90=1 \
    NVCC_THREADS=16 \
    CFLAGS="-I/usr/local/cuda/include/cccl" \
    CXXFLAGS="-I/usr/local/cuda/include/cccl" \
    pip install --no-build-isolation .

# https://gitlab-master.nvidia.com/cudnn/cudnn_frontend/-/tree/devtech/dsa
# git clone ssh://git@gitlab-master.nvidia.com:12051/cudnn/cudnn_frontend.git && cd cudnn_frontend && git checkout devtech/dsa
COPY ./cudnn_frontend/ /opt/cudnn_frontend/
RUN cd /opt/cudnn_frontend/ && \
    pip install --no-build-isolation .
RUN pip install --force-reinstall 'nvidia-cutlass-dsl[cu13]==4.4.2'

# =========================
# Clean cache
# =========================
RUN rm -rf /root/.cache /tmp/*
WORKDIR /home/

# =========================
# Publish the image
# =========================
# docker push gitlab-master.nvidia.com:5005/[repo]:[tag]

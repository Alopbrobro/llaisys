# 使用 NVIDIA CUDA 基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖和构建工具
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 xmake
RUN bash -c "$(curl -fsSL https://xmake.io/shget.text)" && \
    echo 'export PATH=/root/.local/bin:$PATH' >> ~/.bashrc

# 升级 pip 并安装 Python 构建工具
RUN python3 -m pip install --upgrade pip setuptools wheel

# 复制项目文件
COPY . /workspace/llaisys

# 设置工作目录到项目
WORKDIR /workspace/llaisys

# 编译 C++ 库（支持 NVIDIA GPU）
RUN /root/.local/bin/xmake f -m release --nv-gpu=y --root && \
    /root/.local/bin/xmake -j$(nproc) --root && \
    /root/.local/bin/xmake install --root

# 安装 Python 包
RUN cd python && \
    pip install torch>=2.4.0 transformers accelerate && \
    pip install -e .

# 设置默认命令
CMD ["/bin/bash"]

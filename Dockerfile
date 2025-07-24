# SARATR-X Docker镜像构建文件
# 基于PyTorch 1.8.2 + CUDA 11.1 + cuDNN 8构建SAR目标识别基础模型环境

ARG PYTORCH="1.8.2"
ARG CUDA="11.1"
ARG CUDNN="8"

# 使用PyTorch官方镜像作为基础镜像
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

# 设置CUDA编译相关环境变量
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

# 安装系统依赖包
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    wget \
    curl \
    vim \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 首先复制依赖文件以便Docker缓存优化
COPY requirements_pretrain.txt /workspace/
COPY requirements_detection.txt /workspace/
COPY detection/requirements.txt /workspace/detection_req.txt

# 安装Python依赖管理工具
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# 安装支持CUDA的PyTorch
RUN pip install --no-cache-dir torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 \
    --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# 安装预训练所需的timm和tensorboard
RUN pip install --no-cache-dir timm==0.5.4 tensorboard

# 安装检测任务所需的MMCV
RUN pip install --no-cache-dir mmcv-full==1.6.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html

# 安装其他依赖
RUN pip install --no-cache-dir opencv-python

# 安装预训练任务依赖
RUN pip install --no-cache-dir -r requirements_pretrain.txt

# 安装检测任务依赖
RUN pip install --no-cache-dir -r requirements_detection.txt

# 复制项目文件
COPY . /workspace/

# 安装APEX混合精度训练支持
RUN cd /workspace && \
    if [ -f "install_apex.sh" ]; then \
        chmod +x install_apex.sh && \
        ./install_apex.sh; \
    else \
        git clone https://github.com/NVIDIA/apex.git && \
        cd apex && \
        pip install -v --disable-pip-version-check --no-cache-dir \
        --global-option="--cpp_ext" --global-option="--cuda_ext" ./; \
    fi

# 以开发模式安装MMDetection
RUN cd /workspace/detection && \
    pip install --no-cache-dir -e .

# 创建必要的目录
RUN mkdir -p /workspace/data \
    /workspace/output \
    /workspace/checkpoints \
    /workspace/work_dirs

# 设置工具脚本权限
RUN chmod -R +x /workspace/detection/tools/

# 创建入口脚本
RUN echo '#!/bin/bash\n\
# 检查NVIDIA GPU是否可用\n\
if nvidia-smi > /dev/null 2>&1; then\n\
    echo "检测到NVIDIA GPU:"\n\
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv\n\
else\n\
    echo "未检测到NVIDIA GPU或nvidia-smi不可用"\n\
fi\n\
\n\
# 执行提供的命令或启动bash\n\
if [ $# -eq 0 ]; then\n\
    exec /bin/bash\n\
else\n\
    exec "$@"\n\
fi' > /workspace/entrypoint.sh && chmod +x /workspace/entrypoint.sh

# 设置数据集和输出目录的环境变量
ENV DATA_PATH="/workspace/data"
ENV OUTPUT_PATH="/workspace/output"
ENV CHECKPOINT_PATH="/workspace/checkpoints"

# 暴露TensorBoard和Jupyter端口
EXPOSE 6006 8888

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["/bin/bash"]
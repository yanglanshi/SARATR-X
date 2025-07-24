# SARATR-X Docker 配置

本目录包含用于在容器化环境中运行SARATR-X的Docker配置文件。

## 文件说明

- `Dockerfile` - 主要的Docker镜像定义文件
- `docker-compose.yml` - 多服务Docker Compose配置文件
- `docker-run.sh` - 常用Docker操作的实用脚本
- `docker/Dockerfile` - 原始MMDetection专用Dockerfile（遗留文件）

## 快速开始

### 1. 前置条件

- Docker Engine 20.10+
- Docker Compose 1.29+
- NVIDIA Docker运行时（用于GPU支持）
- 支持CUDA 11.1+的NVIDIA GPU

### 2. 设置环境

```bash
# 使实用脚本可执行
chmod +x docker-run.sh

# 检查需求并创建目录
./docker-run.sh setup

# 构建Docker镜像
./docker-run.sh build
```

### 3. 运行服务

```bash
# 启动所有服务（主容器、TensorBoard、Jupyter）
./docker-run.sh start

# 进入主容器
./docker-run.sh shell
```

### 4. 访问点

- **主容器**: `docker-compose exec saratrx bash`
- **TensorBoard**: http://localhost:6007
- **Jupyter Notebook**: http://localhost:8889

## 使用示例

### 预训练

```bash
# 使用默认设置运行预训练
./docker-run.sh pretrain

# 或在容器内手动执行
docker-compose exec saratrx bash
cd /workspace/pre-training
python -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
    --data_path /workspace/data \
    --output_dir /workspace/output
```

### 分类任务

```bash
# 运行分类任务
./docker-run.sh classify

# 或手动执行
docker-compose exec saratrx bash
cd /workspace/classification
bash MIM_linear.sh
```

### 检测任务

```bash
# 运行检测训练
./docker-run.sh detect

# 或手动执行
docker-compose exec saratrx bash
cd /workspace/detection
./tools/dist_train.sh configs/_hivit_/hivit_base_SARDet.py 1
```

## 目录结构

以下目录将自动创建并挂载：

```
SARATR-X/
├── data/           # 训练数据集（挂载点）
├── output/         # 预训练输出
├── checkpoints/    # 模型检查点
├── work_dirs/      # 检测训练输出
└── logs/          # TensorBoard日志
```

## 环境变量

容器中的关键环境变量：

- `DATA_PATH=/workspace/data` - 数据集目录
- `OUTPUT_PATH=/workspace/output` - 输出目录
- `CHECKPOINT_PATH=/workspace/checkpoints` - 检查点目录
- `PYTHONPATH` - 包含所有模块路径

## 数据准备

1. 下载预训练数据集（186K SAR图像）
2. 放入`data/`目录
3. 按ImageFolder结构组织：
   ```
   data/
   └── pretrain/
       ├── class1/
       ├── class2/
       └── ...
   ```

## 模型检查点

1. 下载预训练权重：
   - `mae_hivit_base_1600ep.pth`（ImageNet预训练HiViT）
   - 论文中的SARATR-X检查点
2. 放入`checkpoints/`目录

## GPU配置

设置默认使用单个GPU。对于多GPU配置：

1. 修改`docker-compose.yml`：
   ```yaml
   environment:
     - CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用多个GPU
   ```

2. 更新训练命令以使用多个GPU：
   ```bash
   python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py
   ```

## 故障排除

### GPU未检测到
```bash
# 检查NVIDIA Docker安装
docker run --rm --gpus all nvidia/cuda:11.1-base-ubuntu20.04 nvidia-smi

# 如需要安装nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 内存不足错误
- 减少训练脚本中的批量大小
- 使用梯度累积
- 启用混合精度训练

### 权限问题
```bash
# 修复挂载卷的所有权
sudo chown -R $USER:$USER data/ output/ checkpoints/ work_dirs/ logs/
```

## 实用脚本命令

```bash
./docker-run.sh setup      # 初始设置
./docker-run.sh build      # 构建Docker镜像
./docker-run.sh start      # 启动服务
./docker-run.sh stop       # 停止服务
./docker-run.sh shell      # 进入容器
./docker-run.sh pretrain   # 运行预训练
./docker-run.sh classify   # 运行分类
./docker-run.sh detect     # 运行检测
./docker-run.sh logs       # 显示日志
./docker-run.sh cleanup    # 清理资源
./docker-run.sh help       # 显示帮助
```
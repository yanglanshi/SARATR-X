# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
## 永远优先用中文回答用户问题
## 概述

SARATR-X 是一个SAR（合成孔径雷达）目标识别的基础模型项目，专注于通过自监督学习构建可泛化的表征。该项目包含三个主要模块：预训练、分类和检测。

## 项目结构

- **pre-training/**: 自监督预训练代码，基于SAR-JEPA和HiViT架构
- **classification/**: 少样本学习和线性探测的分类任务代码 
- **detection/**: 基于MMDetection的目标检测代码
- **example/**: 示例文件和可视化结果

## 常用命令

### 预训练
```bash
# 安装依赖
cd pre-training
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install timm==0.5.4 tensorboard
pip install -r requirements_pretrain.txt

# 启动预训练
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --use_env main_pretrain.py \
    --data_path <imagenet-path> --output_dir <pertraining-output-path> \
    --model mae_hivit_base_dec512d6b --mask_ratio 0.75 \
    --batch_size 100 --accum_iter 1 --blr 1.5e-4 --weight_decay 0.05 --epochs 800 --warmup_epochs 5
```

### 分类
```bash
# 基于Dassl的少样本学习
cd classification
# 运行线性探测
bash MIM_linear.sh
```

### 检测
```bash
# 安装检测依赖
cd detection
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html
pip install opencv-python timm==0.5.4
sh ../install_apex.sh
pip install -e .
pip install -r requirements_detection.txt

# 训练检测模型
chmod -R +x tools
./tools/dist_train.sh configs/_hivit_/hivit_base_SARDet.py 8 --work-dir ./work_dirs/SARDet

# 测试模型
python tools/test.py <config_file> <checkpoint_file> --eval bbox
```

## 核心架构

### 预训练模块
- **HiViT Backbone**: 针对SAR图像优化的视觉Transformer架构
- **MAE框架**: 基于掩码自编码器的自监督学习
- **SAR特征提取**: 多尺度梯度特征(GF)，包含9x9、13x13、17x17核大小的特征

### 数据处理
- **预训练数据集**: 180K个未标记的SAR目标样本，涵盖车辆、船舶、飞机等多个类别
- **数据增强**: 支持RandomResizedCrop、RandomHorizontalFlip、ColorJitter等
- **图像尺寸**: 主要为128-512像素，根据不同数据集调整

### 模型配置
- **输入尺寸**: 224x224 (可配置)
- **Patch大小**: 16x16
- **掩码比率**: 0.75
- **编码器**: HiViT-Base架构
- **解码器**: 轻量级重建解码器

## 开发说明

1. **环境要求**: Python 3.9, CUDA 11.1, PyTorch 1.8+
2. **内存需求**: 预训练需要至少32GB GPU内存（8卡并行）
3. **数据格式**: 支持标准ImageFolder格式和自定义SAR数据加载
4. **权重加载**: 支持从ImageNet预训练权重初始化

## 重要文件

- `pre-training/main_pretrain.py`: 预训练主程序
- `pre-training/models/models_hivit_mae.py`: SAR特征的MAE模型实现
- `detection/configs/_hivit_/`: HiViT检测模型配置文件
- `classification/trainers/MIM_linear.py`: 线性探测实现

## 联系方式

如有问题或需要数据集和权重文件，请联系：lwj2150508321@sina.com
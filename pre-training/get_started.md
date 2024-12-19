## Install

### Requirements:
- Python3
- CUDA 11.1
- PyTorch 1.8+ with CUDA support
- timm 0.5.4
- tensorboard

### Step-by-step installation

```bash
conda create -n saratrx python=3.9 -y
conda activate saratrx
cd pre-training

pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install timm==0.5.4 tensorboard
pip install -r requirements.txt

```

## Pre-training
```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --use_env main_pretrain.py
    --data_path <imagenet-path> --output_dir <pertraining-output-path>
    --model mae_hivit_base_dec512d6b --mask_ratio 0.75
    --batch_size 100 --accum_iter 1 --blr 1.5e-4 --weight_decay 0.05 --epochs 800 --warmup_epochs 5 
```


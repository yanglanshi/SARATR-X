<h1 align="center"> SARATR-X: Towards Building A Foundation Model for SAR Target Recognition </h1> 

<h5 align="center"><em> Weijie Li (李玮杰), Wei Yang (杨威), Yuenan Hou (侯跃南), Li Liu (刘丽), Yongxiang Liu (刘永祥), and Xiang Li (黎湘) </em></h5>

<p align="center">
  <a href="#Introduction">Introduction</a> |
  <a href="#Pre-training">Pre-training</a> |
  <a href="#Classification">Classification</a> |
  <a href="#Detection">Detection</a> |
  <a href="#Statement">Statement</a>
</p >

<p align="center">
<a href="https://arxiv.org/abs/2405.09365"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>  
</p>

## Introduction

This is the official repository for the paper “SARATR-X: Towards Building A Foundation Model for SAR Target Recognition”.

这里是论文 “SARATR-X: Towards Building A Foundation Model for SAR Target Recognition (SARATR-X：迈向SAR目标识别基础模型) ”的代码库。

**Abstract:** 
Despite the remarkable progress in synthetic aperture radar automatic target recognition (SAR ATR), recent efforts have concentrated on detecting and classifying a specific category, e.g., vehicles, ships, airplanes, or buildings. One of the fundamental limitations of the top-performing SAR ATR methods is that the learning paradigm is supervised, task-specific, limited-category, closed-world learning, which depends on massive amounts of accurately annotated samples that are expensively labeled by expert SAR analysts and have limited generalization capability and scalability. In this work, we make the first attempt towards building a foundation model for SAR ATR, termed SARATR-X. SARATR-X learns generalizable representations via self-supervised learning (SSL) and provides a cornerstone for label-efficient model adaptation to generic SAR target detection and classification tasks. Specifically, SARATR-X is trained on 0.18 M unlabelled SAR target samples, which are curated by combining contemporary benchmarks and constitute the largest publicly available dataset till now. Considering the characteristics of SAR images, a backbone tailored for SAR ATR is carefully designed, and a two-step SSL method endowed with multi-scale gradient features was applied to ensure the feature diversity and model scalability of SARATR-X. The capabilities of SARATR-X are evaluated on classification under few-shot and robustness settings and detection across various categories and scenes, and impressive performance is achieved, often competitive with or even superior to prior fully supervised, semi-supervised, or self-supervised algorithms. 

**摘要：** 
尽管合成孔径雷达自动目标识别（synthetic aperture radar automatic target recognition, SAR ATR）取得了显著进展，但最近的工作主要集中在对特定类别（如车辆、船舶、飞机或建筑物）的检测和分类上。性能良好的 SAR ATR 方法的一个基本局限是，其学习范式是有监督的、特定任务的、有限类别的、封闭世界的学习，这种学习依赖于大量准确标注的样本，而这些样本是由 SAR 专家分析人员花费高昂成本标注的，其泛化能力和可扩展性有限。在这项工作中，我们首次尝试为 SAR ATR 建立一个基础模型，称为 SARATR-X。SARATR-X 通过自监督学习 (self-supervised learning, SSL) 学习可泛化的表征，为标签高效模型适应通用 SAR 目标检测和分类任务提供了基石。具体来说，SARATR-X 在 0.18 M 个未标记的合成孔径雷达目标样本上进行预训练，这些样本是结合当代数据集基准，构成了迄今为止最大的公开可用预训练数据集。考虑到合成孔径雷达图像的特点，为合成孔径雷达 ATR 量身定制的骨架经过了精心设计，并采用了具有多尺度梯度特征的两步 SSL 方法，以确保 SARATR-X 的特征多样性和模型可扩展性。我们对 SARATR-X 的能力进行了评估，包括少镜头和鲁棒性设置下的分类以及各种类别和场景的检测，其性能令人印象深刻，通常可与之前的全监督、半监督或自监督算法相媲美，甚至更胜一筹。

## Pre-training
Our codes are based on [SAR-JEPA](https://github.com/waterdisappear/SAR-JEPA) and [HiVit](https://github.com/zhangxiaosong18/hivit).

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

### Start Pre-training with SAR images
```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --use_env main_pretrain.py
    --data_path <imagenet-path> --output_dir <pertraining-output-path>
    --model mae_hivit_base_dec512d6b --mask_ratio 0.75
    --batch_size 100 --accum_iter 1 --blr 1.5e-4 --weight_decay 0.05 --epochs 800 --warmup_epochs 5 
```

### Code reading

Q1: How do I use my dataset?

A1: Please change the --data_path and modify the data load code if needed in [main_pretrain.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/main_pretrain.py) and [datasets.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/util/datasets.py).

```python
    # Dataset parameters
    parser.add_argument('--data_path', default='D:\\2023_SARatrX_1\Pre-Train Data\\186K_notest\\', type=str,
                        help='dataset path')
                        
    from util.datasets import load_data
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path), transform=transform_train)
    dataset_train = load_data(os.path.join(args.data_path), transform=transform_train)
    print(len(dataset_train))
```

Q2: How do we make improvements?

A2: You can add more high-quality data and try more data augment methods. Besides, we suggest improvements to the HiViT's attention mechanism in [models_hivit.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/models/models_hivit.py) and our proposed SAR target features in [models_hivit_mae.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/models/models_hivit_mae.py).

```python
    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ])
    
    # SAR feature 
    self.sarfeature1 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=9,
                              img_size=self.img_size,patch_size=self.patch_size)
    self.sarfeature2 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=13,
                              img_size=self.img_size,patch_size=self.patch_size)
    self.sarfeature3 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=17,
                              img_size=self.img_size,patch_size=self.patch_size)
    target = torch.cat([self.patchify(self.sarfeature1(imgs)), self.patchify(self.sarfeature2(imgs)), self.patchify(self.sarfeature3(imgs))], dim=-1)
```
Q3: How to load ImageNet pre-training weights?

A3: You can see in [main_pretrain.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/main_pretrain.py).

```python
    # define the model
    model = models.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    checkpoint = torch.load('./mae_hivit_base_1600ep.pth',
                            map_location='cpu')
    # load pre-trained model
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
```

## Classification

## Detection

## Statement

- If you have any questions or need additional data, code and weight files, please contact us at lwj2150508321@sina.com. 
- If you find our work is useful, please give us 🌟 in GitHub and cite our paper in the following BibTex format:

- 如有任何问题或者需要其他数据、代码和权重文件，请通过 lwj2150508321@sina.com 联系我们。
- 如果您觉得我们的工作有价值，请在 GitHub 上给我们 🌟 并按以下 BibTex 格式引用我们的论文：
```
@article{li2024saratr,
  title={SARATR-X: Towards Building A Foundation Model for SAR Target Recognition},
  author={Li, Weijie and Yang, Wei and Hou, Yuenan and Liu, Li and Liu, Yongxiang and Li, Xiang},
  journal={arXiv preprint},
  url={https://arxiv.org/abs/2405.09365},
  year={2024}
}

@article{li2024predicting,
  title = {Predicting gradient is better: Exploring self-supervised learning for SAR ATR with a joint-embedding predictive architecture},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {218},
  pages = {326-338},
  year = {2024},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2024.09.013},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271624003514},
  author = {Li, Weijie and Yang, Wei and Liu, Tianpeng and Hou, Yuenan and Li, Yuxuan and Liu, Zhen and Liu, Yongxiang and Liu, Li},
}
```

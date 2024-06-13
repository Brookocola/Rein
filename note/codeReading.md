# 源代码阅读笔记
## 代码结构
```
Rein-train
├─checkpoints   模型文件
├─configs   配置代码
│  ├─dinov2
│  ├─dinov2_citys2acdc
│  ├─frozen_vfms
│  └─_base_
│      ├─datasets
│      ├─models
│      └─schedules
├─data  数据集
│  ├─bdd100k
│  │  ├─images
│  │  │  └─10k
│  │  │      ├─test
│  │  │      ├─train
│  │  │      └─val
│  │  └─labels
│  │      └─sem_seg
│  │          ├─masks
│  │          │  ├─train
│  │          │  └─val
│  ├─cityscapes
│  │  ├─gtFine
│  │  │  ├─test
│  │  │  ├─train
│  │  │  └─val
│  │  ├─gtFine_trainvaltest
│  │  ├─leftImg8bit
│  │  │  ├─test
│  │  │  ├─train
│  │  │  └─val
│  │  └─leftImg8bit_trainvaltest
│  ├─gta
│  │  ├─images
│  │  └─labels
│  └─mapillary
│      ├─cityscapes_trainIdLabel
│      │  ├─train
│      │  │  └─label
│      │  └─val
│      │      └─label
│      ├─half
│      │  ├─val_img
│      │  └─val_label
│      ├─testing
│      │  └─v1.2
│      │      ├─images
│      │      ├─instances
│      │      ├─labels
│      │      └─panoptic
│      ├─training
│      │  └─v1.2
│      │      ├─images
│      │      ├─instances
│      │      ├─labels
│      │      └─panoptic
│      └─validation
│          └─v1.2
│              ├─images
│              ├─instances
│              ├─labels
│              └─panoptic
├─note  笔记
│  └─src
├─rein  rein相关代码
│  ├─hooks
│  ├─models
│  │  ├─backbones
│  │  │  ├─dino_layers
│  │  ├─heads
│  │  ├─segmentors
│  ├─optimizers
├─tools 工具代码
│  ├─convert_datasets 数据集转换
│  ├─convert_models 预训练模型转换
│  └─__pycache__
└─work_dirs 结果路径
    ├─rein_dinov2_mask2former_512x512_bs1x4 模型训练
    └─rein_dinov2_mask2former_512x512_bs1x4_test 模型评估测试
```
## 运行流程
### 数据集构建
#### gta数据集
将gta数据集转换成Cityscapes格式（统一同一类别像素值），并统计每张图片中不同类别的像素值个数。
```bash
python tools/convert_datasets/gta.py data/gta 
```
#### cityscapes数据集
转换cityscapes数据集，将json文件转换成png文件，划分训练、验证、测试集
```bash
python tools/convert_datasets/cityscapes.py data/cityscapes
```
#### mapillary数据集
将mapillary数据集转换成Cityscapes格式，并将图像大小设为原始大小的1/2
```bash
python tools/convert_datasets/mapillary2cityscape.py data/mapillary data/mapillary/cityscapes_trainIdLabel --train_id
```
```bash
python tools/convert_datasets/mapillary_resize.py data/mapillary/validation/images data/mapillary/cityscapes_trainIdLabel/val/label data/mapillary/half/val_img data/mapillary/half/val_label
```
#### bd100k数据集
自行下载，无后处理步骤
### 下载转换预训练权重
#### 下载
从[facebookresearch](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)到checkpoints文件夹下
#### 转换
转换预先训练的权重以进行训练或评估  
dinov2：转换dinov2模型以适应输入
```bash
python tools/convert_models/convert_dinov2_large_512x512.py checkpoints/dinov2_vitl14_pretrain.pth
```
### 模型评估
读取配置文件、rein and head和backbone模型（转换后的dinov2模型），在测试集(bdd100k、cityscapes、mapillary)上评估模型精度。
```bash
python tools/test.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py checkpoints/dinov2_rein_and_head.pth --backbone dinov2_converted.pth
```
可以加载已发布的权重文件（和配置文件）评估模型精度。
```bash
python tools/test.py /path/to/cfg /path/to/checkpoint --backbone /path/to/dinov2_converted.pth #(or dinov2_converted_1024x1024.pth)
```
### 模型训练
加载配置文件，训练并评估模型。
```bash
python tools/train.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py
```
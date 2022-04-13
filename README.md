# 基于Paddle复现《Multiresolution Knowledge Distillation for Anomaly Detection》
## 1.简介

无监督表征学习已被证明是图像异常检测/定位的关键组成部分。学习这种表达方式的挑战有两个方面。首先，样本量通常不太大，使用常规技术，不足以学习到足够的通用特征。其次，虽然在训练时只有正常样本可用，但学习的特征应该能够区分正常样本和异常样本。在这里，本文作者使用在ImageNet上预先训练的专家网络在不同层上的特征“蒸馏”到一个更简单的克隆网络来解决这两个问题。在给定输入数据的情况下，我们利用专家网络和克隆网络的中间激活值之间的差异来检测和定位异常。值得注意的是，以前的方法要么无法精确定位异常，要么需要大数据量的基于区域的训练。相比之下，无需任何特殊或有意的训练程序，作者将可解释性算法纳入新框架中，用于异常区域的定位。尽管一些测试数据集和ImageNet之间存在显著的差异，但在异常检测和定位方面与SOTA方法对比，在MNIST、F-MNIST、CIFAR-10、MVTecAD、Retinal-OCT和两个医学数据集上的结果更具有竞争力和显著优势。

<img src=./imgs/anomaly.png></img>

论文地址:

[https://arxiv.org/pdf/2011.11108.pdf](https://arxiv.org/pdf/2011.11108.pdf)

项目地址:

[https://github.com/Niousha12/Knowledge_Distillation_AD](https://github.com/Niousha12/Knowledge_Distillation_AD)


## 2.复现精度

本论文共有两个指标分别是检测与定位指标，一共有15个类别，这里需要对15个类别分别训练，最后取平均值作为验证指标，复现精度如下表所示，其中Paddle行代表的是本次复现精度，可达到论文中的指标。

### Detection test on MVTecAD

| Repo | Bottle | Hazelnut | Capsule | Metal Nut | Leather | Pill | Wood | Carpet | Tile | Grid | Cable | Transistor | Toothbrush | Screw | Zipper | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Origin | 99.39 | 98.37 | 80.46 | 73.58 | 95.05 | 82.7 | 94.29 | 79.25 | 91.57 | 78.01 | 89.19 | 85.55 | 92.17 | 83.31 | 93.24 | 87.74 |
| Paddle | 99.29 | 98.89 | 81.17 | 75.32 | 95.01 | 84.43 | 94.12 | 80.1 | 92.5 | 78.11 | 89.43 | 85.75 | 92.5 | 83.5 | 94.51 | 88.30 |

### Localization test on MVTecAD

| Repo | Bottle | Hazelnut | Capsule | Metal Nut | Leather | Pill | Wood | Carpet | Tile | Grid | Cable | Transistor | Toothbrush | Screw | Zipper | Mean  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |-------|
| Origin | 96.32 | 94.62 | 95.86 | 86.38 | 98.05 | 89.63 | 84.8 | 95.64 | 82.77 | 91.78 | 82.4 | 76.45 | 96.12 | 95.96 | 93.9 | 90.71 |
| Paddle | 96.20 | 94.32 | 96.12 | 86.59 | 98.06 | 89.98 | 85.29 | 95.69 | 82.87 | 91.84 | 83.46 | 76.49 | 96.02 | 96.20 | 94.91 | 90.94 |

Detection test的Mean为88.30。

Localization test的Mean为90.94。

达到验收指标。


## 3.数据集

下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/116034](https://aistudio.baidu.com/aistudio/datasetdetail/116034)

下载数据集以后，执行以下命令解压数据集。

```shell
mkdir MVTec
cd MVTec
tar xvf ../mvtec_anomaly_detection.tar.xz
```
这样数据集就解压到MVTec目录下了。

VGG16模型权重下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/137835](https://aistudio.baidu.com/aistudio/datasetdetail/137835)

下载之后，需要将权重文件拷贝到源码的根目录下。

```shell
cp vgg16.pdparams Knowledge_Distillation_AD_Paddle/vgg16.pdparams
```

15个模型权重下载地址:

链接: [https://pan.baidu.com/s/1ItWP6mXEZe3GPFErYKaLag](https://pan.baidu.com/s/1ItWP6mXEZe3GPFErYKaLag)

提取码: bnwi 




## 4.环境依赖
PaddlePaddle == 2.2.0

## 5.快速开始
MVTec数据集中共有15个类别，每个类别都需要单独训练一个模型，在训练时，通过normal_class参数来指定数据进行训练。

### 模型训练

训练命令如下：

```shell
nohup python -u train.py --config configs/config.yaml --dataset_root ../data/MVTec/ --normal_class toothbrush > logs/toothbrush_train.log

# 查看训练日志
tail -f toothbrush_train.log
```
参数说明:

config: 配置文件路径

dataset_root: 数据集路径

normal_class: 参与训练的数据类别名称，共有15种，类别名称参考复现精度表格中的名称。15个类别为:bottle、capsule、grid、leather、metal_nut、tile、transistor、zipper、cable   carpet、hazelnut、pill、screw、toothbrush、wood。

训练模型日志，都保存在logs目录中，以<class_name>_train.log格式命名，<class_name>对应的类别的名称。

训练结束后，会保存指标最优模型在output/<class_name>目录下。

下面是toothbrush类别训练过程的部分日志供参考。

```shell
[Train] epoch [0/601], loss:8.4637 class:toothbrush
[Eval] toothbrush class RocAUC at epoch 0: 0.4
[Eval] save best model at epoch 0
[Train] epoch [1/601], loss:8.1224 class:toothbrush
[Train] epoch [2/601], loss:7.7782 class:toothbrush
[Train] epoch [3/601], loss:7.4917 class:toothbrush
[Train] epoch [4/601], loss:7.2435 class:toothbrush
[Train] epoch [5/601], loss:7.0313 class:toothbrush
[Train] epoch [6/601], loss:6.8482 class:toothbrush
[Train] epoch [7/601], loss:6.6858 class:toothbrush
[Train] epoch [8/601], loss:6.5398 class:toothbrush
[Train] epoch [9/601], loss:6.4067 class:toothbrush
[Train] epoch [10/601], loss:6.2836 class:toothbrush
[Eval] toothbrush class RocAUC at epoch 10: 0.4028
[Eval] save best model at epoch 10
[Train] epoch [11/601], loss:6.1681 class:toothbrush
[Train] epoch [12/601], loss:6.0595 class:toothbrush
[Train] epoch [13/601], loss:5.9577 class:toothbrush
[Train] epoch [14/601], loss:5.8617 class:toothbrush
[Train] epoch [15/601], loss:5.7718 class:toothbrush
[Train] epoch [16/601], loss:5.6858 class:toothbrush
[Train] epoch [17/601], loss:5.6049 class:toothbrush
[Train] epoch [18/601], loss:5.5266 class:toothbrush
[Train] epoch [19/601], loss:5.4520 class:toothbrush
[Train] epoch [20/601], loss:5.3802 class:toothbrush
[Eval] toothbrush class RocAUC at epoch 20: 0.4056
```
### 模型验证

本项目提供了test.py脚本来验证模型精度，执行该脚本会计算detection test和localization test。

命令如下：

```shell
python test.py --config configs/config.yaml --dataset_root ../data/MVTec --normal_class toothbrush --model_path ./output/toothbrush/best_model.pdparams
```

参数说明:

config: 配置文件路径

dataset_root: 数据集路径

normal_class: 参与训练的数据类别名称，共有15种，类别名称参考复现精度表格中的名称。

model_path: 模型路径

以toothbrush类别为例，输出的日志如下:

```shell
Loading pretrained model from output/toothbrush/best_model.pdparams
There are 45/45 variables loaded into VGG.
Vanilla Backpropagation:
mvtec: toothbrush class localization test RocAUC: 0.9602441191673279
mvtec: toothbrush class detection test RocAUC: 0.925

```

完整的验证日志保存在logs目录下，以<class_name>_val.log格式命名，<class_name>对应的类别的名称。



### 单张图片预测
本项目提供了单张图片的预测脚本，可生成标注异常部分的图片。使用方法如下：
```shell
python predict.py --image_path carpet.png --model_path pretrained_weights/carpet/best_model.pdparams --save_dir ./output --threshold 0.5
```

参数说明：

image_path:需要预测的图片

model_path: 模型路径

save_dir: 输出图片保存路径

threshold: 热力图显示阈值，默认为0.5，只有大于0.5值才会在热力图上显示。

预测样例:

<div>
<img src=./imgs/capsule.png width=256></img>
<img src=./imgs/capsule_pred.png width=256></img>
</div>


<div>
<img src=./imgs/carpet.png width=256></img>
<img src=./imgs/carpet_pred.png width=256></img>
</div>

### 模型导出
模型导出可执行以下命令：

```shell
python export_model.py --config configs/config.yaml --model_path ./output/capsule/best_model.pdparams --save_dir ./output/
```

参数说明：

config: 配置文件路径

model_path: 模型路径

save_dir: 输出图片保存路径

### Inference推理

可使用以下命令进行模型推理。但由于本论文中推理预测生成heatmap的方法都需要计算梯度，所以Inference并没有实际的使用意义。如果希望生成heatmap需要使用上面提到的predict.py脚本。本脚本的意义在于可以跑通tipc，运行命令如下：

```shell
python infer.py
--use_gpu=False --enable_mkldnn=False --cpu_threads=2 --model_file=./test_tipc/output/model.pdmodel --batch_size=2 --input_file=test_tipc/data/MVTec/capsule/test/good --enable_benchmark=True --precision=fp32 --params_file=./test_tipc/output/model.pdiparams
```

参数说明:

use_gpu:是否使用GPU

enable_mkldnn:是否使用mkldnn

cpu_threads: cpu线程数
 
model_file: 模型路径

batch_size: 批次大小

input_file: 输入文件路径

enable_benchmark: 是否开启benchmark

precision: 运算精度

params_file: 模型权重文件，由export_model.py脚本导出。

### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://github.com/LDOUBLEV/AutoLog
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
```


```shell
bash test_tipc/prepare.sh test_tipc/configs/KDAD/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/KDAD/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示：

<img src=./test_tipc/data/tipc_result.png></img>

## 6.代码结构与详细说明

```
Knowledge_Distillation_AD_Paddle
├── README.md # 用户指南
├── configs # 配置文件
├── dataloader.py # 数据读取器
├── dataset.py # 数据集
├── export_model.py #模型导出
├── imgs # 图片资源
├── infer.py # 模型推理脚本
├── logs # 训练与验证日志
├── loss_functions.py # 损失函数
├── network.py # 网络结构
├── param_init.py # 参数初始化 
├── test.py # 测试脚本
├── test_functions.py # 测试函数
├── test_tipc # TIPC 基础测试链条
├── train.py # 训练脚本
├── utils # 工具链
├── vgg.py # VGG模型架构 专家模型
└── vgg16.pdparams # VGG在ImageNet上的权重

```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| KDAD |
|框架版本| PaddlePaddle==2.2.0|
|应用场景| 工业质检 |


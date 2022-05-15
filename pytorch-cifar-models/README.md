# Experiments on CIFAR datasets with PyTorch

## Introduction
Reimplement state-of-the-art CNN models in cifar dataset with PyTorch, now including:

1.[ConvNeXt](https://arxiv.org/abs/2201.03545)

2.[ResNet](https://arxiv.org/abs/1512.03385v1)

3.Other derivatives including: [PreActResNet](https://arxiv.org/abs/1603.05027v3), [WideResNet](https://arxiv.org/abs/1605.07146v4), [ResNeXt](https://arxiv.org/abs/1611.05431v2), [DenseNet](https://arxiv.org/abs/1608.06993v4)

other results will be added later.

## Requirements:software
Requirements for [PyTorch](http://pytorch.org/)

## Requirements:hardware
For most experiments, one NVIDIA RTX 3090 with 24G memory is enough.

## Usage
1. Clone this repository

```
git clone https://github.com/Evergreen0929/Mid-Term-Project.git  
cd ./pytorch-cifar-models
```

In this project, the network structure is defined in the models folder, the script ```gen_mean_std.py``` is used to calculate
the mean and standard deviation value of the dataset.

2. Edit main.py and run.sh

In the ```main.py```, you can specify the network you want to train(for example):

```
model = resnet20_cifar(num_classes=100)
...
fdir = 'result/convnext_small_cifar100'
```

Then, you need specify some parameter for training in ```run.sh```. For covnext:

```
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 64 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct 100
```

Additionally, the pretraining of MAE and train with the initialization can be achieved by setting the main_recon.py:

```
CUDA_VISIBLE_DEVICES=0 python main_recon.py --epoch 240 --batch-size 64 --lr 0.05 --momentum 0.9 --wd 1e-4 -ct 100 --aug_ratio 1 --alpha 0.5 --train_ae_epoch 120
```

3. Train

```
nohup sh run.sh > convnext_small_cifar100.log &
```

After training, the training log will be recorded in the .log file, the best model(on the test set) 
will be stored in the fdir.

**Note**:For first training, cifar10 or cifar100 dataset will be downloaded, so make sure your comuter is online.
Otherwise, download the datasets and decompress them and put them in the ```data``` folder.

4. Test

```
CUDA_VISIBLE_DEVICES=0 python main.py -e --resume=fdir/model_best.pth.tar
```

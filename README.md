<div align="center">    
 
# CIFAR100: Train models with pytorch lightning and timm 
</div>


## Install dependencies   
```bash
cd LitCifar100
# install project   
conda create -y -n ICCV python=3.10
conda activate ICCV
conda install pytorch=1.13.1 torchvision cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
 ```


 ## Training on CIFAR100 Dataset
Next, navigate to LitCifar100 folder and run `train.py`. It may take **2~3** hours on **4** NVIDIA 2080Ti gpus using ddp.

```bash
# train DenseNet121 model using 4 gpus and ddp on CIFAR100 dataset
python -u train.py --config config/densenet.yaml
```
For single gpu user, you may change the `densenet.yaml` file to
```yaml
gpus: 1
```

```bash
# finetune VitB/16 model using 4 gpus and ddp on CIFAR100 dataset
python -u train.py --config config/vit.yaml
```

```bash
# train resnet model using 4 gpus and ddp on CIFAR100 dataset, resnetxxx need to be replaced by specific model names,like resnet18.
python -u train.py --config config/resnetxxx.yaml
```




## Experimental Results 

### DenseNet121
| 序号  |     模型      | Acc@1 | Augmentation | Label Smoothing | Img Size | Model<br>Initial |    Scheduler     |
| :-: | :---------: | :---: | :----------: | :-------------: | :------: | :--------------: | :--------------: |
|  0  | DenseNet121 | 67.6  |     Auto     |        ×        |    32    |  Official Init   |      StepLR      |
|  1  | DenseNet121 | 67.91 |     Auto     |        √        |    32    |  Official Init   |      StepLR      |
|  2  | DenseNet121 | 76.34 |     Auto     |        √        |    64    |  Official Init   |      StepLR      |
|  3  | DenseNet121 | 75.22 |     Auto     |        √        |    64    |  Official Init   | Reduce on Plateu |
|  4  | DenseNet121 | 75.45 |     Auto     |        √        |    64    |   Xaiver Init    |      StepLR      |
|  5  | DenseNet121 | 78.06 |     Auto     |        √        |    64    |  Official Init   |  Warmup+Cosine   |
|  6  | DenseNet121 | 78.57 | Auto + mixup |        √        |    64    |  Official Init   |  Warmup+Cosine   |
|  7  | DenseNet121 | 81.68 |     Auto     |        √        |   128    |  Official Init   |  Warmup+Cosine   |
|  8  | DenseNet121 |       | Auto + mixup |        √        |   128    |  Official Init   |  Warmup+Cosine   |

### ResNet
|    模型    | Acc@1 | Augmentation | Label Smoothing | Img Size | Model<br>Initial | Scheduler |
| :------: | :---: | :----------: | :-------------: | :------: | :--------------: | :-------: |
| ResNet18 | 71.21 |     Auto     |        √        |    64    |  Official Init   |  StepLR   |
| ResNet34 | 72.97 |     Auto     |        √        |    64    |  Official Init   |  StepLR   |
| ResNet50 | 74.86 |     Auto     |        √        |    64    |  Official Init   |  StepLR   |

### ViT
|   模型    | Acc@1 | Augmentation | Label Smoothing | Img Size | Model<br>Initial | Scheduler |
| :-----: | :---: | :----------: | :-------------: | :------: | :--------------: | :-------: |
| ViTB/16 | 89.5  |     Auto     |        √        |   224    |  Official Init   |  StepLR   |
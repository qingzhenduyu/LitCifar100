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
Next, navigate to LitCifar100 folder and run `train.py`. It may take **2~3** hours on **3** NVIDIA 2080Ti gpus using ddp.

```bash
# train DenseNet121 model using 3 gpus and ddp on CROHME dataset
python -u train.py --config config/densenet.yaml
```

For single gpu user, you may change the `densenet.yaml` file to
```yaml
gpus: 1
```



## Implementation Details
- CIFAR100

|   epoch   | learning rate |  weight decay | Optimizer | Momentum |  Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:---------:|
|   0 ~ 60  |      0.1      |     5e-4    |    SGD    |    0.9   |   False   |
|  61 ~ 120 |      0.01     |     5e-4     |    SGD    |    0.9   |   False   |
| 121 ~ 180 |      0.001    |     5e-4     |    SGD    |    0.9   |   False   |
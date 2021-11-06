# Grouped Channel Pruning: A Group-and-Prune Paradigm for Compressing Convolutional Neural Networks

## Requirements
- Python 3.6
- PyTorch 1.0.0
- TorchVision 0.2.0
- tqdm

## Usage of CIFAR-10 
Three-stage operations: (1) pretraining (optional), (2) training+regularizing, (3) pruning+fine-tuning

**Pretraining**

```
# pretrain ResNet20
python cifar_pretrain.py -l 20 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]

# pretrain ResNet32
python cifar_pretrain.py -l 32 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]

# pretrain ResNet56
python cifar_pretrain.py -l 56 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]

# pretrain ResNet110
python cifar_pretrain.py -l 110 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]
```

**Training+regularizing**

```
# retrain ResNet20
python cifar_train.py -l 20 [--reg 5e-5] [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]

# retrain ResNet32
python cifar_train.py -l 32 [--reg 3e-5] [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]

# retrain ResNet56
python cifar_train.py -l 56 [--reg 2e-5] [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]

# retrain ResNet110
python cifar_train.py -l 110 [--reg 1e-5] [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]
```

**Pruning+finetuning**

```
# prune 60% channels in ResNet20 with 16 groups
python cifar_prune.py -l 20 -g 16 -c 0.4 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune 70% channels in ResNet20 with 16 groups
python cifar_prune.py -l 20 -g 16 -c 0.3 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune 60% channels in ResNet20 with 4 groups
python cifar_prune.py -l 20 -g 4 -c 0.4 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune 60% channels in ResNet32 with 16 groups
python cifar_prune.py -l 32 -g 16 -c 0.4 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune 60% channels in ResNet56 with 16 groups
python cifar_prune.py -l 56 -g 16 -c 0.4 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune 60% channels in ResNet110 with 16 groups
python cifar_prune.py -l 110 -g 16 -c 0.4 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]
```


## Usage of ImageNet 

Import gcp.py and replace the SGD optimizer in your training code.

```
import gcp

...

# SGD optimizier
optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# SGD with group lasso optimizier
optimizer = SGD_GL(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# SGD + GCP
optimizer = SGD_GCP(cnn.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                        ratio=args.comp, numgroup=args.groups)

...

# In training with regularization, add a group lasso term to loss
loss = criterion(outputs, labels) + optimizer.reg(reg, 5e-5)

...

```

## Hyperparameters of Models for Imagenet

ResNet18: alpha = 5e-5
ResNet50: alpha = 3e-5
MobileNetV2: alpha = 5e-5


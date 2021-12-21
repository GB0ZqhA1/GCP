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
python cifar_train.py -l 20 -g 8 --reg 5e-5 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# retrain ResNet32
python cifar_train.py -l 32 -g 8 --reg 3e-5 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# retrain ResNet56
python cifar_train.py -l 56 -g 8 --reg 2e-5 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# retrain ResNet110
python cifar_train.py -l 110 -g 8 --reg 1e-5 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]
```

**Pruning+finetuning**

```
# prune ResNet20 with 8 maximum groups and delta=0.3
python cifar_prune.py -l 20 -g 8 -t 0.3 --reg 5e-5 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune ResNet20 with 8 maximum groups and delta=0.4
python cifar_prune.py -l 20 -g 8  -t 0.4 --reg 5e-5 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune ResNet20 with 4 maximum groups and delta=0.3
python cifar_prune.py -l 20 -g 4 -t 0.3 --reg 5e-5 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune ResNet32 with 8 maximum groups and delta=0.3
python cifar_prune.py -l 32 -g 8 -t 0.3 --reg 3e-5 [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune ResNet56 with 8 maximum groups and delta=0.3
python cifar_prune.py -l 56 -g 8 -t 0.3 --reg 2e-5  [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]

# prune ResNet110 with 8 maximum groups and delta=0.3
python cifar_prune.py -l 110 -g 8 -t 0.3 --reg 1e-5  [--save-dir ./cifarmodel] [--workers 4] [--epochs 164] [--batch-size 128] [--lr 0.05] [--momentum 0.9] [--wd 1e-3]
```




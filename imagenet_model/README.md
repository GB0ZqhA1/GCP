# Accelerated models for ImageNet inference

Accelerated models for ImageNet inference compiled by Pytorch JIT for CUDA.

## Requirements
- Python 3.6
- PyTorch 1.10.0
- TorchVision 0.11.0
- tqdm


**Usage**
```
model = torch.jit.load('optimized_pruned_resnet18.pt')
images = torch.randn(batch_size,3,224,224).cuda()
output = model(images)
```

import argparse
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from gcp import *
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from inf_resnet import *

warnings.filterwarnings("ignore")

#########################################
executionID = "0"
comp = 0.25
groups = 4
rampath = '/media/ramdisk/'
########################################
os.makedirs(rampath, exist_ok=True)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-m', '--model', metavar='MODEL', default=None,
                    help='model file')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    global comp
    global groups
    loadname = args.model
    
    model = models.__dict__[args.arch]()
    model.cuda()
    for m in model.modules():
        if isinstance(m, models.resnet.BasicBlock):
            m.conv1.weight.position=0
            m.conv2.weight.position=1
        if isinstance(m, models.resnet.Bottleneck):
            m.conv1.weight.position=0
            m.conv2.weight.position=1
            m.conv3.weight.position=2
    optimizer = SGD_GCP(model.parameters(), lr=0, momentum=0, weight_decay=0,
                        ratio=comp, numgroup=4)
    if os.path.isfile(loadname):
        print("=> loading checkpoint '{}'".format(loadname))
        # Map model to be loaded to specified single gpu.
        loc = 'cuda'
        checkpoint = torch.load(loadname, map_location=loc)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[15:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}'"
                .format(loadname))
    else:
        print("=> no checkpoint found at '{}'".format(loadname))
        return

    optimizer.cluster()

    if args.arch == "resnet18":
        inf_model = inf_resnet18(groups=groups, comp=comp).cuda()
    elif args.arch == "resnet50":
        inf_model = inf_resnet50(groups=groups, comp=comp).cuda()
    model.eval()
    inf_model.eval()

    for m1, m2 in zip(model.modules(),inf_model.modules()):
        if isinstance(m1, models.resnet.BasicBlock) or isinstance(m1, models.resnet.Bottleneck) or\
            isinstance(m1, models.resnet.ResNet):
            m2.load_from_pruned_checkpoints(m1)
    inf_model = torch.jit.optimize_for_inference(torch.jit.trace(inf_model, torch.rand(64,3,224,224).cuda()))
    
    print('Results with acceleration')
    validate(inf_model, args)
    print('Results without acceleration (Same speed to the original model)')
    validate(model, args)
    
    return

def validate(model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # switch to evaluate mode
    with torch.no_grad():
        for i in range(100):
            images = torch.randn(64,3,224,224).cuda()

            # compute output
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(images)
            end.record()
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(start.elapsed_time(end))
            
        print(' * Time {batch_time.avg:.3f}'
              .format(batch_time=batch_time))

    return 0

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
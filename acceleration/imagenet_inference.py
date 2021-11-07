import argparse
import os
import random
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataload import *
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
path = '/media/ssd/epoch/'
path2 = '/media/ssd2/epoch/'
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
best_acc1 = 0


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    global best_acc1
    global comp
    global groups
    filename = args.arch
    loadname = args.model
    exc = ProcessPoolExecutor(max_workers=1)
    for file in os.listdir(rampath):
        if file.endswith(".hdf5"):
            os.remove(rampath+file)
    exc.submit(shutil.copy, './testset.hdf5', rampath+'testset.hdf5')
    testpath = rampath+'testset.hdf5'

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
        for k, v in checkpoint['state_dict'].items():
            name = k[15:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                .format(loadname, checkpoint['epoch'], checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(loadname))

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

    val_dataset = HdfDataset(testpath, False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    print('Results with acceleration')
    validate(val_loader, inf_model, args)
    print('Results without acceleration (Same speed to the original model)')
    validate(val_loader, model, args)
    
    return

def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    prep = tofloat()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = prep(images.cuda(non_blocking=True))
                target = target.cuda(non_blocking=True)

            # compute output
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(images)
            end.record()
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(start.elapsed_time(end))
            #loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))


            # if i % args.print_freq == 0:
            #     progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Time {batch_time.avg:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(batch_time=batch_time,top1=top1, top5=top5))

    return top1.avg

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
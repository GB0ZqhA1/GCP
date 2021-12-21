import argparse
import os
import warnings
from gcp import *
from torchvision.models.resnet import BasicBlock, Bottleneck

import torch
import torch.nn.parallel
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from inf_model import *

warnings.filterwarnings("ignore")

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
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

def set_groups(network,groups,inds):
    for m in network.modules():
        if isinstance(m, BasicBlock):
            m.conv2.weight.groups = groups[0]
            m.conv1.weight.groups = groups[1]
            m.conv2.weight.ind = inds[0]
            m.conv1.weight.ind = inds[1]
            if len(groups)>1:
                groups = groups[2:]
                inds = inds[2:]
                
        elif isinstance(m, Bottleneck):
            m.conv3.weight.groups = groups[0]
            m.conv2.weight.groups = groups[1]
            m.conv1.weight.groups = groups[2]
            m.conv3.weight.ind = inds[0]
            m.conv2.weight.ind = inds[1]
            m.conv1.weight.ind = inds[2]
            if len(groups)>1:
                groups = groups[3:]
                inds = inds[3:]


def main():
    args = parser.parse_args()
    main_worker(args)


loc = 'cuda'

def main_worker(args):
    global comp
    global groups
    loadname = args.model
    
    org_model = models.__dict__[args.arch](pretrained=True)
    org_model.to(loc)
    if loadname is None:
        model = org_model
    else:
        model = models.__dict__[args.arch]()
        model.to(loc)
        if os.path.isfile(loadname):
            print("=> loading checkpoint '{}'".format(loadname))
            checkpoint = torch.load(loadname, map_location=loc)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[15:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                    .format(loadname, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(loadname))

    groups = torch.load(args.arch+'R_8x_0.4_groups.pkl', map_location=torch.device(loc))
    inds = torch.load(args.arch+'R_8x_0.4_inds.pkl', map_location=torch.device(loc))
    pruner = GCP(model, 0, 0, 8)
    set_groups(model, groups, inds)
    if args.arch == "resnet18":
        inf_model = inf_resnet18().to(loc)
    elif args.arch == "resnet50":
        inf_model = inf_resnet50().to(loc)

    for m1, m2 in zip(model.modules(),inf_model.modules()):
        if isinstance(m1, models.resnet.BasicBlock) or isinstance(m1, models.resnet.Bottleneck) or\
            isinstance(m1, models.resnet.ResNet):
            m2.load_from_pruned_checkpoints(m1)
    
    model.eval()
    inf_model.eval()

    args.batch_size=128
    input_tensor = torch.randn(128,3,224,224).to(loc)
    inf_model = torch.jit.optimize_for_inference(torch.jit.trace(inf_model, input_tensor))
    print('Speed of the original model')
    validate(org_model, args, count=100)
    print('Speed of the accelerated model')
    validate(inf_model, args, count=100)
    #cudabench(inf_model, input_tensor)
    torch.jit.save(inf_model, 'pruned_'+args.arch+'.pt')
    return

def cudabench(model, input_tensor, count=100):
    with torch.no_grad():
        with torch.autograd.profiler.profile(use_cuda=True, with_flops=True) as prof:
            for _ in range(count):
                output = model(input_tensor)
        print(prof.key_averages().table(sort_by="self_cuda_time_total"))

def validate(model, args, count=100000, verbose=True):
    batch_time = AverageMeter('Time', ':6.3f')    
    with torch.no_grad():
        for _ in range(count):
        
            images = torch.randn(args.batch_size,3,224,224).to(loc)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(images)
            end.record()
            torch.cuda.synchronize()
            batch_time.update(start.elapsed_time(end))
        
    
        if verbose:
            print(' * Time {batch_time.avg:.3f}'.format(batch_time=batch_time))

    return

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

if __name__ == '__main__':
    main()

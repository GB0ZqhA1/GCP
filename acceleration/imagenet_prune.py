import argparse
import os
import shutil
import time
import warnings
from gcp import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

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
parser.add_argument('-m', '--model', metavar='MODEL', default='',
                    help='model file')
parser.add_argument('-d', '--delta', metavar='ARCH', default=0.3,
                    help='value of delta')


def main():
    args = parser.parse_args()
    main_worker(1, args)


def main_worker(gpu, args):
    global best_acc1
    global comp
    delta = args.delta
    filename = args.arch
    loadname = args.model
    groupname = filename + 'R_8x_'+str(delta)+'_groups.pkl'
    indname = filename + 'R_8x_'+str(delta)+'_inds.pkl'
    args.gpu = gpu

    # create model

    model = models.__dict__[args.arch](pretrained=True)
    
    # optionally resume from a checkpoint
    if os.path.isfile(loadname):
        print("=> loading checkpoint '{}'".format(loadname))
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(loadname, map_location=loc)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[15:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        best_acc1 = checkpoint['best_acc1']
        print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                .format(loadname, checkpoint['epoch'], checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(loadname))
        
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    pruner = GCP(model, 0, delta, 8)
    pruner.initialize()

    torch.save(pruner.groups, groupname)
    torch.save(pruner.inds, indname)


if __name__ == '__main__':
    main()

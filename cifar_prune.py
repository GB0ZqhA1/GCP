import os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from gcp import *
from tqdm import tqdm
from model import *
import argparse
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description='CIFAR-10 Pruning')
parser.add_argument('--save_dir', type=str, default='./cifarmodel/', help='Folder to save checkpoints and log.')
parser.add_argument('-a', '--arch', default='resnet', type=str, metavar='N', help='network architecture (default: resnet)')
parser.add_argument('-l', '--layers', default=20, type=int, metavar='N', help='number of ResNet layers (default: 20)')
parser.add_argument('-d', '--device', default='0', type=str, metavar='N', help='main device (default: 0)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=164, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--comp', default=-1, type=float, metavar='N', help='remaining channels (%)')
parser.add_argument('-t', '--threshold', default=-1, type=float, metavar='N', help='remaining channels (%)')
parser.add_argument('--reg', default=-1, type=float, metavar='N', help='remaining channels (%)')
parser.add_argument('-g', '--groups', default=8, type=int, metavar='N', help='number of groups (default: 4)')
parser.add_argument('--mode', type=str, default='AO', help='Pruning mode (C or AO) (default: AO)')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.device

def warmup(optimizer, lr, epoch):
    if epoch < 2:
        lr = lr/4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if epoch == 2:
        lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]

device = torch.device("cuda")

def train(filename, network):

    train_dataset = dsets.CIFAR10(root='./dataset',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                           std=(0.2470, 0.2435, 0.2616))
                                  ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True, drop_last=True)
    test_dataset = dsets.CIFAR10(root='./dataset',
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                          std=(0.2470, 0.2435, 0.2616))
                                 ]))
    # Data Loader (Input Pipeline)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, num_workers=args.workers,
                                              shuffle=False)

    torch.backends.cudnn.benchmark=True
    cnn, netname = network(args.layers)
    config = netname+"_%d"%(args.groups)
    
    if 'resnet' in filename:
        loadpath = args.save_dir+'/resnet%d_R%s_%.5f_%dx.pkl'%(args.layers, args.device, args.reg, args.groups)
    elif 'wrn' in filename:
        loadpath = args.save_dir+'/wrn%d_R%s_%.5f_%dx.pkl'%(args.layers, args.device, args.reg, args.groups)
    state_dict, baseacc = torch.load(loadpath)
    print(loadpath)
    cnn.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()


    bestacc=0
    if 'resnet' in filename:
        optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif 'wrn' in filename:
        optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    scheduler = CosineAnnealingLR(optimizer, args.epochs)
    pruner = GCP(cnn, args.comp, args.threshold, args.groups)
    totalloss, layer_wise_cr = pruner.initialize()
    pruner.prune()
    rp = 0
    tp = 0
    comps = []
    total = []
    for m in cnn.modules():
        if isinstance(m, nn.Conv2d):
            param = m.weight
            r = (param.data == 0).sum().item()
            t = param.data.numel()
            rp += r
            tp += t
            if hasattr(param, "comp_rate"):
                o,i,k,_ = param.size()
                comps.append(i/o*k*k*param.comp_rate)
                total.append(i/o*k*k)
    flop_rate = sum(comps)/sum(total)*100
    print("total parameters : %d/%d (%f)" % (tp - rp, tp, (tp - rp) / tp))

    bar = tqdm(total=len(train_loader) * args.epochs)
    for epoch in range(args.epochs):
        cnn.train()
        warmup(optimizer, args.lr, epoch)
        bar.set_description("["+config+"]CH:%.2f|BASE:%.2f|ACC:%.2f" % (flop_rate, baseacc, bestacc))
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            gpuimg = images.to(device)
            labels = labels.to(device)

            outputs = cnn(gpuimg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pruner.prune()
            bar.update()

        scheduler.step()

        cnn.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum().item()
        acc = 100 * correct / total
        cnn.train()

        if bestacc<acc:
            bestacc=acc
            torch.save([cnn.state_dict(),bestacc,flop_rate], args.save_dir+filename)
            bar.set_description(
                "[" + config + "]CH:%.2f|BASE:%.2f|ACC:%.2f" % (flop_rate, baseacc, bestacc))

    bar.close()
    return bestacc, flop_rate, totalloss, layer_wise_cr

def resnet(layers):
    return CifarResNet(ResNetBasicblock, layers, 10).to(device), "resnet"+str(layers)

def wrn(layers):
    return Wide_ResNet(layers, 8, 10).to(device), "wrn"+str(layers)


if __name__ == '__main__':
    results = []
    if args.arch == 'resnet':
        results.append(train('resnet%d_C%s_%.5f_%dx.pkl'%(args.layers, args.device, args.reg, args.groups), resnet))

    elif args.arch == 'wrn':
        results.append(train('wrn%d_C%s_%.5f_%dx.pkl'%(args.layers, args.device, args.reg, args.groups), wrn))
    

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

parser = argparse.ArgumentParser(description='CIFAR-10 Retraining')
parser.add_argument('--save_dir', type=str, default='./cifarmodel/', help='Folder to save checkpoints and log.')
parser.add_argument('-a', '--arch', default='resnet', type=str, metavar='N', help='network architecture (default: resnet)')
parser.add_argument('-l', '--layers', default=-1, type=int, metavar='N', help='number of ResNet layers (default: 20)')
parser.add_argument('-d', '--device', default='0', type=str, metavar='N', help='main device (default: 0)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=164, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--comp', default=0.25, type=float, metavar='N', help='remaining channels (%)')
parser.add_argument('-g', '--groups', default=8, type=int, metavar='N', help='number of groups (default: 4)')
parser.add_argument('-r', '--reg', type=float, metavar='R', help='Group lasso hyperparameter (default: auto)')

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
    reg = args.reg
    

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
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, num_workers=args.workers,
                                              shuffle=False)

    torch.backends.cudnn.benchmark=True
    cnn, netname = network(args.layers)
    config = netname
    if 'resnet' in filename:
        loadpath = args.save_dir+'/resnet%d_%s.pkl'%(args.layers, args.device)
    elif 'wrn' in filename:
        loadpath = args.save_dir+'/wrn%d_%s.pkl'%(args.layers, args.device)
    print(loadpath)
    cnn.load_state_dict(torch.load(loadpath)[0])

    criterion = nn.CrossEntropyLoss()
    bestacc=0
    
    if 'resnet' in filename:
        optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif 'wrn' in filename:
        optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    scheduler = CosineAnnealingLR(optimizer, args.epochs)
    pruner = GCP(cnn, args.comp, 0, args.groups)
    pruner.initialize(False)
    bar = tqdm(total=len(train_loader) * args.epochs)
    for epoch in range(args.epochs):
        cnn.train()
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            gpuimg = images.to(device)
            labels = labels.to(device)

            outputs = cnn(gpuimg)
            loss = criterion(outputs, labels) + prune_reg(cnn, reg)
            loss.backward()
            optimizer.step()
            bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f" % (get_lr(optimizer)[0], loss.item(), bestacc))
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
            torch.save([cnn.state_dict(),bestacc], args.save_dir+filename)
            bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f" % (get_lr(optimizer)[0], loss.item(), bestacc))
    bar.close()
    return bestacc

def resnet(layers):
    return CifarResNet(ResNetBasicblock, layers, 10).to(device), "resnet"+str(layers)

def wrn(layers):
    return Wide_ResNet(layers, 8, 10).to(device), "wrn"+str(layers)


if __name__ == '__main__':
    results = []
    if args.arch == 'resnet':
        results.append(train('resnet%d_R%s_%.5f_%dx.pkl'%(args.layers, args.device, args.reg, args.groups), resnet))
        
    elif args.arch == 'wrn':
        results.append(train('wrn%d_R%s_%.5f_%dx.pkl'%(args.layers, args.device, args.reg, args.groups), wrn))
        
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
parser.add_argument('-l', '--layers', default=20, type=int, metavar='N', help='number of ResNet layers (default: 20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=164, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--comp', type=float, metavar='N', help='remaining channels (%)')
parser.add_argument('-g', '--groups', default=4, type=int, metavar='N', help='number of groups (default: 4)')
parser.add_argument('--mode', type=str, default='AO', help='Pruning mode (C or AO) (default: AO)')

args = parser.parse_args()

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
    config = netname+"%d_%d"%(args.layers,args.groups)
    state_dict, baseacc = torch.load(args.save_dir+'/'+ netname+ '.pkl')

    cnn.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()


    bestacc=0
    optimizer = SGD_GCP(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                        ratio=args.comp, numgroup=args.groups)

    if args.mode == "C":
        optimizer.kmeans()
    elif args.mode == "AO":
        optimizer.cluster()
    else:
        print("Invalid pruning mode")
        return

    scheduler = CosineAnnealingLR(optimizer, args.epochs)

    rp = 0
    tp = 0
    comps = []
    for m in cnn.modules():
        if isinstance(m, ResNetBasicblock):
            m.conv_a.weight.nextind=m.conv_b.weight.ind
            m.conv_b.weight.prevind = m.conv_a.weight.ind
    for m in cnn.modules():
        if isinstance(m, nn.Conv2d):
            param = m.weight
            r = (param.data == 0).sum().item()
            t = param.data.numel()
            rp += r
            tp += t
            if hasattr(param, "nextind"):
                comp = param.ind.float().mean().item() * (param.nextind.float().sum(0) > 0).float().mean().item()
                comps.append(comp)
            elif hasattr(param, "ind"):
                comp = param.ind.float().mean().item()
                comps.append(comp)
    print("total parameters : %d/%d (%f)" % (tp - rp, tp, (tp - rp) / tp))

    bar = tqdm(total=len(train_loader) * args.epochs)
    for epoch in range(args.epochs):
        cnn.train()
        bar.set_description("["+config+"]CH:%.2f|BASE:%.2f|ACC:%.2f" % (sum(comps)/len(comps)*100, baseacc, bestacc))
        for step, (images, labels) in enumerate(train_loader):

            gpuimg = images.to(device)
            labels = labels.to(device)

            outputs = cnn(gpuimg)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
            torch.save([cnn.state_dict(),bestacc,sum(comps)/len(comps)], args.save_dir+filename)
            bar.set_description(
                "[" + config + "]CH:%.2f|BASE:%.2f|ACC:%.2f" % (sum(comps) / len(comps) * 100, baseacc, bestacc))

    bar.close()
    return baseacc, bestacc, sum(comps)/len(comps)

def network(layers):
    return CifarResNet(ResNetBasicblock, layers, 10).to(device), "resnet"+str(layers)


if __name__ == '__main__':
    train('resnet%d%.2fC%dx.pkl'%(args.layers,args.comp,args.groups), network)

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
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
parser.add_argument('-l', '--layers', default=20, type=int, metavar='N', help='number of ResNet layers (default: 20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=164, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--reg', default=-1, type=float, metavar='R', help='Group lasso hyperparameter (default: auto)')

args = parser.parse_args()

def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]

device = torch.device("cuda")

def train(filename, network):
    reg = args.reg
    if reg==-1:
        if args.layers == 20:
            reg = 5e-5
        elif args.layers == 32:
            reg = 3e-5
        elif args.layers == 56:
            reg = 2e-5
        elif args.layers == 110:
            reg = 1e-5
        else:
            print("Set group lasso hyperparameter manually")
            return

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
    cnn.load_state_dict(torch.load(args.save_dir+'/'+netname+'.pkl')[0])

    criterion = nn.CrossEntropyLoss()
    bestacc=0
    optimizer = SGD_GL(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, [args.epochs//2,args.epochs*3//4], 0.1)

    bar = tqdm(total=len(train_loader) * args.epochs)
    for epoch in range(args.epochs):
        cnn.train()
        for step, (images, labels) in enumerate(train_loader):
            gpuimg = images.to(device)
            labels = labels.to(device)

            outputs = cnn(gpuimg)
            loss = criterion(outputs, labels) + optimizer.reg(reg)

            optimizer.zero_grad()
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

def network(layers):
    return CifarResNet(ResNetBasicblock, layers, 10).to(device), "resnet"+str(layers)

if __name__ == '__main__':
    train('resnet%dR.pkl'%(args.layers), network)
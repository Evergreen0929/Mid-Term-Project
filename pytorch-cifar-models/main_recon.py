import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchtoolbox.transform import Cutout
from torchtoolbox.tools import mixup_data, mixup_criterion
from cutmix import cutmix_data
from cutout import cutout_data

from models.convnext_decoder import ConvNeXt_recon, convnext_small
from models.resnet_cifar_decoder import resnet164_cifar

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='100', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
#parser.add_argument('--aug_choice', default='None', type=str, choices=('None', 'mixup', 'cutout', 'cutmix', 'cutout_recon'), help='choices for data augmentation')
parser.add_argument('--alpha', default=0.8, type=float, help='ratio of mixture')
parser.add_argument('--aug_ratio', default=0.5, type=float, help='ratio of augmented data')
parser.add_argument('--recon', default='True', type=str)
parser.add_argument('--cutout', default='True', type=str)
parser.add_argument('--train_ae_epoch', default=50, type=int)
parser.add_argument('--pretrain_model', default=False, type=bool)


best_prec = 0

def main():
    global args, best_prec
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Model building
    print('=> Building model...')
    if use_gpu:
        # model can be set to anyone that I have defined in models folder
        # note the model should match to the cifar type !

        #model = resnet20_cifar()
        # model = resnet32_cifar()
        # model = resnet44_cifar()
        # model = resnet110_cifar()
        # model = preact_resnet110_cifar()

        # model = create_RepLKNet31T(num_classes=100)
        # model = convnext_small(num_classes=100, pretrained=False, in_22k=False)

        model = resnet164_cifar(num_classes=100)
        #model = convnext_small(num_classes=100, pretrained=True, in_22k=True)
        # model = resnet1001_cifar(num_classes=100)
        # model = preact_resnet164_cifar(num_classes=100)
        # model = preact_resnet1001_cifar(num_classes=100)

        # model = wide_resnet_cifar(depth=26, width=10, num_classes=100)

        # model = resneXt_cifar(depth=29, cardinality=16, baseWidth=64, num_classes=100)
        
        #model = densenet_BC_cifar(depth=190, k=40, num_classes=100)

        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = 'result/convnextsmall_cifar100_recon{}_cutout{}_alpha{}_pretrein_new'.format(args.recon, args.cutout, args.alpha)
        print("save dir: {}".format(fdir))
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        writer = SummaryWriter(log_dir=fdir)

        # adjust the lr according to the model type
        model_type = 4

        # dummy_input = torch.rand(1, 3, 32, 32)
        # writer.add_graph(model, (dummy_input,))
        # del dummy_input

        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        #optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        cudnn.benchmark = True

    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.pretrain_model == False:
                args.start_epoch = checkpoint['epoch']
                best_prec = checkpoint['best_prec']
                optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # CIFAR100
    else:
        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        # train_dataset = torchvision.datasets.CIFAR100(
        #     root='./data',
        #     train=True,
        #     download=True,
        #     transform=transforms.Compose([
        #         transforms.RandomCrop(32, padding=4),
        #         Cutout(p=args.aug_ratio),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        validate(testloader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, model_type)

        # train for one epoch
        train(trainloader, model, criterion, optimizer, epoch, writer)

        # evaluate on test set
        prec = validate(testloader, model, criterion, writer)
        if epoch >= args.train_ae_epoch:
            writer.add_scalar('test_acc', prec, epoch)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)

    print("\nBest Acc: {:.4f}".format(best_prec))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

class DeNormalize(object):
    def __init__(self, mean_std=([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])):
        self.mean, self.std = mean_std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def up_sampling(x, ratio):
    return F.interpolate(x, scale_factor=ratio, mode='bilinear')

def train(trainloader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cutout == 'True' and np.random.rand() < args.aug_ratio:
            mask_input = cutout_data(input, args.alpha)
        else:
            mask_input = input
        mask_input = mask_input.cuda()
        input, target = input.cuda(), target.cuda()

        if epoch < args.train_ae_epoch:
            _, recon = model(mask_input)
            recon_loss = nn.L1Loss()(up_sampling(recon, 4), up_sampling(input, 4))
            loss = torch.Tensor([0])
            prec = torch.Tensor([0])
        else:
            output, _ = model(input)
            loss = criterion(output, target)
            recon_loss = torch.Tensor([0])
            prec = accuracy(output, target)[0]

        losses.update(loss.item(), input.size(0))
        recon_losses.update(recon_loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        if epoch < args.train_ae_epoch:
            loss = recon_loss * 0.01
        else:
            loss = loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'REconLoss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, recon_loss=recon_losses ,top1=top1))

    if epoch < args.train_ae_epoch:
        writer.add_scalar('train_loss_recon', recon_losses.avg, epoch)
        writer.add_image('image', DeNormalize()(input[0]), epoch)
        writer.add_image('mask_image', DeNormalize()(mask_input[0]), epoch)
        writer.add_image('recon_image', DeNormalize()(recon.detach().clone()[0]), epoch)
    else:
        writer.add_scalar('train_loss', losses.avg, epoch)
        writer.add_scalar('train_acc', top1.avg, epoch)


def validate(val_loader, model, criterion, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output, recon = model(input)
            loss = criterion(output, target)
            recon_loss = nn.L1Loss()(recon, input)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            recon_losses.update(recon_loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'REconLoss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, recon_loss=recon_losses,
                   top1=top1))
                writer.add_image('val/image', DeNormalize()(input[0]), i)
                writer.add_image('val/recon_image', DeNormalize()(recon.detach().clone()[0]), i)

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, model_type):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if model_type == 1:
        if epoch < 80:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif model_type == 2:
        if epoch < 60:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.2
        elif epoch < 160:
            lr = args.lr * 0.04
        else:
            lr = args.lr * 0.008
    elif model_type == 3:
        if epoch < 150:
            lr = args.lr
        elif epoch < 225:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif model_type == 4:
        lr = args.lr * (1 - epoch / (args.epochs + 1)) ** 0.9
    elif model_type == 5:
        if epoch < 120:
            lr = args.lr * (1 - epoch / (args.epochs + 1 - 120)) ** 0.9
        else:
            lr = args.lr * (1 - (epoch - 120) / (args.epochs + 1 - 120)) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    main()


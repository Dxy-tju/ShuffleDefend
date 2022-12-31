import argparse
from email.policy import default
import os
import shutil
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from img_data.tiny_imgnet import TinyImageNet
from torch.autograd import Variable
from wideresnet import WideResNet
from filternet6 import FilterNet
from tensorboardX import SummaryWriter
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
from attack import fgsm_whitebox

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', 
                                # default='cifar10', 
                                # default='svhn',
                                default='mnist',
                                type=str,
                                help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default= 10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,    # default=0.01
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=34, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', 
                                # default='./runs/fnet_basic/fnet_type2.pth.tar', 
                                default='',
                                type=str,
                                help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='fnet_basic', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)


args = parser.parse_args()
best_loss = 10
best_prec1 = 0

def main():
    global args, best_loss, best_prec1
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
            ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        # normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    # assert(args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data',
                                                # split='train' , 
                                                train=True, 
                                                download=True,
                                                transform=transform_train),
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', 
                                                # split='test' ,
                                                train=False, 
                                                download=True, 
                                                transform=transform_test),
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                **kwargs)

    # train_dataset = TinyImageNet('train', transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    # val_dataset = TinyImageNet('val', transform=transform_test)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # create model
    model = FilterNet(input_nbr=1, gaussian_rate=0.075, net_mode_train=True)
    model_test = WideResNet(args.layers, 10, args.widen_factor, dropRate=args.droprate)
    checkpoint_test = torch.load('./runs/wrn_mnist_best.pth.tar')
    model_test.load_state_dict(checkpoint_test['state_dict'])

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    model_test = model_test.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # cosine learning rate
    writer = SummaryWriter(log_dir='log_dir',comment='test_tensorboard')
    
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, optimizer, epoch, writer)

        # evaluate on validation set
        prec1, cur_loss = validate(val_loader, model, model_test, criterion, epoch)


        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_loss = min(cur_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best)
    print('Best accuracy: ', best_prec1)
    print('Best loss: ', best_loss)
    writer.close()

def train_loss(input, output):
    return ((output - input) ** 2).mean()

def train(train_loader, model, optimizer, epoch, writer):
    
    model.netmode = True
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
                
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        
        # compute output
        output = model(input)
        
        loss = train_loss(input, output)

        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)


def validate(val_loader, model, model_test, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    adv_top1 = AverageMeter()

    # switch to evaluate mode
    model.netmode = False
    model.eval()
    model_test.eval()
    

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        adv_input = fgsm_whitebox(model_test, input, target)

        # compute output
        with torch.no_grad():
            generate = model(input)
            output = model_test(generate)
            adv_output = model_test(model(adv_input))

        loss = train_loss(input, generate)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        adv_prec1 = accuracy(adv_output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        adv_top1.update(adv_prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  'Adv_Prec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, adv_top1=adv_top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * Adv_Prec@1 {adv_top1.avg:.3f}'.format(adv_top1=adv_top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='fnet_type2.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'fnet_type2_best.pth.tar')

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


if __name__ == '__main__':
    main()

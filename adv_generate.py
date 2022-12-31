from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
# from models.wideresnet import *
from wideresnet import WideResNet
# from models.resnet import *
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=8.0/255.0,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2.0/255.0,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='runs/WideResNet-34-10/wrn_best.pth.tar',
                    # default='./runs/WideResNet-34-10/targeted_trades.pth.tar',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--attack-method', default='CW')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
args = parser.parse_args()
print(args)

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
transform_test = transforms.Compose([
        transforms.ToTensor(),
        # normalize
        ])

# set up data loader
if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'svhn':
    args.epsilon = 4.0 / 255.0
    args.step_size = 1.0 / 255.0
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
else:
    raise NotImplementedError

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor



def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
       
    return X_pgd-X


def main():

    if args.white_box_attack:
        # white-box attack
        print('white-box attack')
        
        model = WideResNet(depth=34, widen_factor=10, num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
       
        pgd_noise = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            X, y = Variable(data, requires_grad=True), Variable(target)
            # print(X)
            adv_img = _pgd_whitebox(model, X, y)
            adv_img = adv_img.detach().cpu().numpy()
            adv_img = np.squeeze(adv_img)
            pgd_noise.append(adv_img[0])
            pgd_noise.append(adv_img[1])
            pgd_noise.append(adv_img[2])
            print('One minibatch done, adv_img =', adv_img)
        np.save('./pgd_noise.npy', pgd_noise)
        np.savetxt('./pgd_noise.txt', adv_img, delimiter=" ")
        
if __name__ == '__main__':
    main()
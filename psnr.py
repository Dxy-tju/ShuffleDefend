import numpy as np
import math
import argparse
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
from filternet6 import FilterNet
from wideresnet import WideResNet
from attack import fgsm_whitebox, pgd_whitebox
from PIL import Image
from tensorboardX import SummaryWriter
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--generated-path', default='./timgnet_super_resolution_img', type=str,
                    help='path to save generated imgs')
parser.add_argument('--clean-path', default='./timgnet_clean_img', type=str,
                    help='path to save clean imgs')
parser.add_argument('--adv-path', default='./timgnet_pgd_img', type=str,
                    help='path to save adversarial generated imgs')
parser.add_argument('--resume', default='./runs/fnet_basic/fnet_type6_timgnet_best.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.set_defaults(augment=True)


args = parser.parse_args()

def psnr(img1, img2):
   img1 = img1.data.cpu().numpy()
   img2 = img2.data.cpu().numpy()
   mse = np.mean( (img1 - img2) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def main():
    
    args = parser.parse_args()

    # Data loading code

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
        transforms.ToTensor(),
        # normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    val_dataset = TinyImageNet('val', transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # create model
    model = FilterNet(input_nbr=3, gaussian_rate=1, net_mode_train=False)
    model_test = WideResNet(34, 200, 10, 0)
    checkpoint_test = torch.load('./runs/wrn_timgnet_best.pth.tar')
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
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        print("Best loss =", best_loss)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        return

    cudnn.benchmark = True

    psnr_fgsm = 0
    psnr_pgd = 0
    psnr_gaussian = 0
    psnr_fgsm_filtered = 0
    psnr_pgd_filtered = 0
    psnr_gaussian_filtered = 0
    psnr_clean_filtered = 0
    psnr_clean = 0
    total = 0
    for i, (input, target) in enumerate(val_loader):
       target = target.cuda(non_blocking=True)
       input = input.cuda(non_blocking=True)
        
       model.eval()
       model_test.eval()
       
      #  fgsm_img = fgsm_whitebox(model_test, input, target)
      #  pgd_img = pgd_whitebox(model_test, input, target)
       noise = torch.randn(input.size()).cuda() * 0.075
       gaussian_img = torch.clamp(torch.add(input, noise), 0, 1)
       
      #  fgsm_filtered = model(fgsm_img)
      #  pgd_filtered = model(pgd_img)
       gaussian_filtered = model(gaussian_img)
       
      #  psnr_fgsm += psnr(input, fgsm_img)
      #  psnr_pgd += psnr(input, pgd_img)
       psnr_gaussian += psnr(input, gaussian_img)
      #  psnr_fgsm_filtered += psnr(input, fgsm_filtered)
      #  psnr_pgd_filtered += psnr(input, pgd_filtered)
       psnr_gaussian_filtered += psnr(input, gaussian_filtered)
    #    psnr_clean_filtered += psnr(input, model(input))
    #    psnr_clean += psnr(input, input)
       total += 1
       print('Calculating psnr, current pic =', total)
    
    # print('psnr_clean_filtered = ', psnr_clean_filtered/total)
    # print('psnr_clean = ', psnr_clean/total)

   #  print('psnr_fgsm = ', psnr_fgsm/total)
   #  print('psnr_pgd = ', psnr_pgd/total)
    print('psnr_gaussian = ', psnr_gaussian/total)
   #  print('psnr_fgsm_filtered = ', psnr_fgsm_filtered/total)
   #  print('psnr_pgd_filtered = ', psnr_pgd_filtered/total)
    print('psnr_gaussian_filtered = ', psnr_gaussian_filtered/total)


if __name__ == '__main__':
    main()

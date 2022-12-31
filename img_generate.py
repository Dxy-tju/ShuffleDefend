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
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
    #                      transform=transform_train),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)

    val_dataset = TinyImageNet('val', transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # create model
    model = FilterNet(input_nbr=3, gaussian_rate=1, net_mode_train=False)
    # model_test = WideResNet(34, 200, 10, 0)
    checkpoint_test = torch.load('./runs/fnet_basic/fnet_type6_timgnet_best.pth.tar')
    # model_test.load_state_dict(checkpoint_test['state_dict'])

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    # model_test = model_test.cuda()

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

    if not os.path.exists(args.generated_path):
            os.makedirs(args.generated_path)
    if not os.path.exists(args.clean_path):
            os.makedirs(args.clean_path)
    if not os.path.exists(args.adv_path):
            os.makedirs(args.adv_path)
    if not os.path.exists('./gaussian_cifar10'):
            os.makedirs('./gaussian_cifar10')
    if not os.path.exists('./clean_cifar10'):
            os.makedirs('./clean_cifar10')
    if not os.path.exists('./gaussian_filtered_timgnet'):
            os.makedirs('./gaussian_filtered_timgnet')
    for i, (input, target) in enumerate(val_loader):
                
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        
        model.eval()
        # model_test.eval()
        
        # generated_img, super_resolution = model(input)
        # adv_img = pgd_whitebox(model_test, input, target)
        # generated_img = model(pgd_whitebox(model_test, input, target))
        
        # generated_img_path = "{}/generated_{}.jpg".format(args.generated_path, i)
        # clean_img_path = "{}/clean_{}.jpg".format(args.clean_path, i)
        # adv_img_path = "{}/adv_{}.jpg".format(args.adv_path, i)
        
        # save_image(generated_img_path, super_resolution)
        # save_image(generated_img_path, generated_img)
        # save_image(clean_img_path, input)
        # save_image(adv_img_path, adv_img)
        # noise_path = './gaussian_noise.jpg'
        # noise_img_path = './gaussian_noised_img.jpg'
        # filtered_img_path = './gaussian_denoised_img.jpg'
        # origin_path =  './gaussian_origin_img.jpg'
        
        # gaussian_feature_path = "{}/feature_{}.jpg".format('./gaussian_cifar10', i)
        # clean_feature_path = "{}/clean_{}.jpg".format('./clean_cifar10', i)
        filtered_feature_path = "{}/filtered_{}.jpg".format('./gaussian_filtered_timgnet', i)
        # noise = torch.randn(input.size()).cuda() * 0.075
        # gaussian_img = torch.add(input, noise)
        # gaussian_img = torch.clamp(gaussian_img, 0, 1)
        noise = torch.randn(input.size()).cuda() * 0.075
        gaussian_pic = torch.add(noise, input)
        gaussian_pic = torch.clamp(gaussian_pic, 0, 1)
        gaussian_feature = gaussian_pic
        clean_feature = input
        filtered_feature = model(gaussian_feature)
       
        # save_image(noise_path, noise)
        # save_image(noise_img_path, gaussian_img)
        # save_image(filtered_img_path, filtered_img)
        # save_image(origin_path, input)
        # save_image(gaussian_feature_path, gaussian_feature)
        # save_image(clean_feature_path, clean_feature)
        save_image(filtered_feature_path, filtered_feature)
        # break
        if(i%100==0 and i!=0):
            print("All img generated")
            break
            

def save_image(save_path, generated_img):
    img = generated_img.data.cpu().numpy()
    img = img.transpose(0, 2, 3, 1) * 255.0
    img = np.squeeze(img)
    img = np.array(img).astype(np.uint8)
    img_pil = Image.fromarray(img)
    img_pil.save(save_path)

if __name__ == '__main__':
    main()

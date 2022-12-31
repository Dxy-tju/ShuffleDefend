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
from torch.autograd import Variable
from filternet3 import FilterNet
from resnet import resnet56
from wideresnet import WideResNet
from attack import fgsm_whitebox
from PIL import Image
from tensorboardX import SummaryWriter
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value


def save_image(save_path, generated_img):
    img = generated_img.data.cpu().numpy()
    img = img.transpose(1, 2, 0) * 255.0
    img = np.squeeze(img)
    img = np.array(img).astype(np.uint8)
    img_pil = Image.fromarray(img)
    img_pil.save(save_path)
    
model = resnet56()
path = './runs/resnet_best.pth.tar'

print("=> loading checkpoint '{}'".format(path))
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
model.cuda()


transform_test = transforms.Compose([
        transforms.ToTensor(),
        # normalize
        ])
kwargs = {'num_workers': 1, 'pin_memory': True}
val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data',
                         train=False,
                         transform=transform_test),
                         batch_size=1,
                         shuffle=True, 
                         **kwargs)



index_list1 = [0, 7, 15]
index_list2 = [0, 15, 31]
index_list3 = [0, 31, 63]

if not os.path.exists('./resnet_pic'):
            os.makedirs('./resnet_pic')
            
if not os.path.exists('./resnet_feature1_clean'):
            os.makedirs('./resnet_feature1_clean')
if not os.path.exists('./resnet_feature2_clean'):
            os.makedirs('./resnet_feature2_clean')
if not os.path.exists('./resnet_feature3_clean'):
            os.makedirs('./resnet_feature3_clean')
            
if not os.path.exists('./resnet_feature1_adv'):
            os.makedirs('./resnet_feature1_adv')
if not os.path.exists('./resnet_feature2_adv'):
            os.makedirs('./resnet_feature2_adv')
if not os.path.exists('./resnet_feature3_adv'):
            os.makedirs('./resnet_feature3_adv')

for i, (input, target) in enumerate(val_loader):

        model.eval()
        
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        adv_img = fgsm_whitebox(model, input, target)
        
        _, feature1, feature2, feature3 = model(input)
        _, feature1_adv, feature2_adv, feature3_adv = model(adv_img)
        
        save_image('./resnet_pic/origin_pic.jpg', input[0])
        
        for index in index_list1:
                save_img_path = "{}/feature1_clean_{}.jpg".format('./resnet_feature1_clean', index)
                save_image(save_img_path, torch.unsqueeze(feature1[index], dim=0))
                adv_img_path = "{}/feature1_adv_{}.jpg".format('./resnet_feature1_adv', index)
                save_image(adv_img_path, torch.unsqueeze(feature1_adv[index], dim=0))
        
        for index in index_list2:
                save_img_path = "{}/feature2_clean_{}.jpg".format('./resnet_feature2_clean', index)
                save_image(save_img_path, torch.unsqueeze(feature2[index], dim=0))
                adv_img_path = "{}/feature2_adv_{}.jpg".format('./resnet_feature2_adv', index)
                save_image(adv_img_path, torch.unsqueeze(feature2_adv[index], dim=0))
        
        for index in index_list3:
                save_img_path = "{}/feature3_clean_{}.jpg".format('./resnet_feature3_clean', index)
                save_image(save_img_path, torch.unsqueeze(feature3[index], dim=0))
                adv_img_path = "{}/feature3_adv_{}.jpg".format('./resnet_feature3_adv', index)
                save_image(adv_img_path, torch.unsqueeze(feature3_adv[index], dim=0))
        
        print("All features generated")
        break

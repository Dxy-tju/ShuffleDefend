import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FilterNet(nn.Module):
    def __init__(self,input_nbr=3, gaussian_rate = 1, net_mode_train=True):
        super(FilterNet, self).__init__()

        batchNorm_momentum = None
        self.gaussian_rate = gaussian_rate
        self.netmode = net_mode_train

        #stage 1
        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 12, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(12, momentum= batchNorm_momentum)
        self.sigmoid = nn.Sigmoid()
        
        #PixelShuffle
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        #stage 2
        self.conv21 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
        self.bn21 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv23 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn23 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        #stage 2d
        self.conv23d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn23d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(3, momentum= batchNorm_momentum)


    def forward(self, x):
        
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        
        #PixelShuffle
        x12_shuffle = self.pixel_shuffle(x12)

        # Stage 2
        x12_shuffle = self.sigmoid(x12_shuffle)
        
        noise = torch.randn(x12_shuffle.size()).cuda() * self.gaussian_rate
        if(self.netmode):
            x12_shuffle = torch.add(x12_shuffle, noise)
            x12_shuffle = torch.clamp(x12_shuffle, 0, 1)
        # np.save('feature.npy', x12_shuffle.detach().cpu().numpy())
        
        x21 = F.relu(self.bn21(self.conv21(x12_shuffle)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x23 = F.relu(self.bn23(self.conv23(x22)))
       
        # Stage 2d

        x23d = F.relu(self.bn23d(self.conv23d(x23)))
        x22d = F.relu(self.bn22d(self.conv22d(x23d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        return x21d


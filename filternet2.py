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
        self.conv11 = nn.Conv2d(input_nbr, 32, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(32, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        
        #stage 2
        # self.conv21 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2) 
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.atroconv21 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.atrobn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        
        self.conv22 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.atroconv22 = nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2)
        self.atrobn22 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        
        # self.conv23 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2)
        self.bn23 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.atroconv23 = nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2)
        self.atrobn23 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        
        #stage 3
        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        #stage 3d
        self.conv32d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        #stage 2d
        self.conv23d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn23d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        
        self.conv22d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        #stage 1d
        self.conv12d = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(32, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.bn11d = nn.BatchNorm2d(3, momentum= batchNorm_momentum)


    def forward(self, x):
        
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x12)))
        feature21 = x21
        x22 = F.relu(self.bn22(self.conv22(x21)))
        feature22 = x22
        x23 = F.relu(self.bn23(self.conv23(x22)))
        feature23 = x23

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x23)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        
        # Stage perturbation
        noise = torch.randn(x32.size()).cuda() * self.gaussian_rate
        if(self.netmode):
            x32 = torch.add(x32, noise)
            
        # Stage 3d        
        x32d = F.relu(self.bn32d(self.conv32d(x32)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        # x31d = torch.add(x31d, F.relu(self.atrobn23(self.atroconv23(feature23))))
        x31d = torch.add(x31d, feature23)
        x23d = F.relu(self.bn23d(self.conv23d(x31d)))
        x23d = torch.add(x23d, feature22)
        x22d = F.relu(self.bn22d(self.conv22d(x23d)))
        x22d = torch.add(x22d, feature21)
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x12d = F.relu(self.bn12d(self.conv12d(x21d)))
        x11d = self.conv11d(x12d)

        return x11d


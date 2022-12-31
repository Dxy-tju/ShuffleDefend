import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms

val_labels = np.load('./img_data/val_labels.npy')
images = np.load('./img_data/image.npy')
val_images = np.load('./img_data/val_image.npy')

class TinyImageNet(Dataset):
    def __init__(self, type, transform):
        self.type = type
        if type == 'train':
            self.images = np.load('./img_data/image.npy')
        elif type == 'val':
            self.val_images = np.load('./img_data/val_image.npy')
        self.transform = transform
        
    def __getitem__(self, index):
        if self.type == 'train':
            label = index//500
            image = self.images[index]
        elif self.type == 'val':
            label = val_labels[index]
            image = self.val_images[index]
        return self.transform(image), label
        
    def __len__(self):
        len = 0
        if self.type == 'train':
            len = self.images.shape[0]
        if self.type == 'val':
            len = self.val_images.shape[0]
        return len
import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms

labels_t = []
image_names = []
with open('./tiny-imagenet-200/wnids.txt') as wnid:
    for line in wnid:
        labels_t.append(line.strip('\n'))
for label in labels_t:
    txt_path = './tiny-imagenet-200/train/'+label+'/'+label+'_boxes.txt'
    image_name = []
    with open(txt_path) as txt:
        for line in txt:
            image_name.append(line.strip('\n').split('\t')[0])
    image_names.append(image_name)
labels = np.arange(200)

val_labels_t = []
val_labels = []
val_names = []
with open('./tiny-imagenet-200/val/val_annotations.txt') as txt:
    for line in txt:
        val_names.append(line.strip('\n').split('\t')[0])
        val_labels_t.append(line.strip('\n').split('\t')[1])
for i in range(len(val_labels_t)):
    for i_t in range(len(labels_t)):
        if val_labels_t[i] == labels_t[i_t]:
            val_labels.append(i_t)
val_labels = np.array(val_labels)
np.save('val_labels.npy', val_labels)
print('Val label loaded')

timg_images = []
timg_val_images = []

i = 0
for label in labels_t:
    image = []
    for image_name in image_names[i]:
        image_path = os.path.join('./tiny-imagenet-200/train', label, 'images', image_name) 
        image.append(cv2.imread(image_path))
    timg_images.append(image)
    i = i + 1
timg_images = np.array(timg_images)
timg_images = timg_images.reshape(-1, 64, 64, 3)


for val_image in val_names:
    val_image_path = os.path.join('./tiny-imagenet-200/val/images', val_image)
    timg_val_images.append(cv2.imread(val_image_path))
timg_val_images = np.array(timg_val_images)
np.save('image.npy', timg_images)
np.save('val_image.npy', timg_val_images)
print("All data loaded")
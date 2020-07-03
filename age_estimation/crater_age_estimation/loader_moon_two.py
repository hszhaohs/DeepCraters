import os
import numpy as np
import random
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T

class MoonData(data.Dataset):
    
    def __init__(self, imglist, attrlist, labels=None, attr_range=(0,78), transforms=None):
        self.imgs = imglist
        self.attr = attrlist
        self.attr_range = attr_range
        if labels is not None:
            self.labels = np.array(labels)
        else:
            imgs_num = len(imglist)
            self.labels = np.zeros(imgs_num)
        
        if transforms is None:
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.ToTensor(),
                normalize
                ])
        else:
            self.transforms = transforms
        
    def __getitem__(self, index):
        """
        一次返回一条数据
        """
        img_path = self.imgs[index]
        attr_path = self.attr[index]
        label = self.labels[index]
        img = Image.open(img_path)
        img = self.transforms(img)
        img1 = torch.Tensor(img.shape).copy_(img)
        attr = np.loadtxt(open(attr_path), delimiter=',')
        attr = torch.Tensor(attr[self.attr_range[0]:self.attr_range[1]])
        attr1 = torch.Tensor(attr.shape).copy_(attr)
        return img, attr, label, img1, attr1
        
    def __len__(self):
        return len(self.imgs)


def horizontal_flip(image, rate=0.5):
    if random.random() < rate:
        #image = np.flip(image,1).copy()
        image = image[:, ::-1, :]
    return image

def random_crop(image, crop_size, padding=4):
    crop_size = check_size(crop_size)
    image = np.pad(image,((padding,padding),(padding,padding),(0,0)),'constant',constant_values=0)
    h, w, _ = image.shape
    top = random.randrange(0, h - crop_size[0])
    left = random.randrange(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


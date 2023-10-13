import os
import numpy as np
import random
from glob import glob
import h5py
from typing import Optional, Tuple, Union, List
import warnings
warnings.simplefilter('ignore')

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, distributed
from torchvision import transforms
import torchvision.transforms.functional as F

class ACDCDatasets(Dataset):
    def __init__(self,
                 base_dir,
                 split='train',
                 transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform

        if self.split == 'train':
            self.sample_list = glob(self.base_dir + '/train/*.h5')
        else:
            self.sample_list = glob(self.base_dir + '/val/*.h5')
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(self.sample_list[idx], 'r')
        image = h5f['image'][()].astype(np.float32)
        label = h5f['label'][()].astype(np.float32)

       # h, w = image.shape
       # if h != w:
       #     if h < w:
       #         pad_width = (((w-h)//2, w-h-(w-h)//2), (0, 0))
       #     else:
       #         pad_width = ((0, 0), ((h-w)//2, h-w-(h-w)//2))
       #     
       #     image = np.pad(image, pad_width, mode='reflect')
       #     label = np.pad(label, pad_width, mode='reflect')

        sample = {'image': image, 'label':label}
        if self.transform:
            sample = self.transform(sample)
        sample['idx'] = idx
        return sample

class ToPILImage():
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # convert to uint8 type for PIL convertion
        image = (image*255.).astype(np.uint8)

        image = F.to_pil_image(image)
        label = F.to_pil_image(label)
        return {'image': image, 'label': label}
    
class ToTensor(transforms.ToTensor):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.to_tensor(image)
        label = F.to_tensor(label)
        return {'image': image, 'label': label}

class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

class Resize(nn.Module):
    def __init__(self, size,
                interpolation=F.InterpolationMode.NEAREST, 
                max_size=None, antialias="warn" ):
        super().__init__()
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias
    
    def forward(self, sample):
        image, label = sample['image'], sample['label']
        image = F.resize(image, self.size, self.interpolation, self.max_size,
                         self.antialias)
        label = F.resize(label, self.size, self.interpolation, self.max_size,
                         self.antialias)
        return {'image': image, 'label': label}

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, sample):
        if random.random() < self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            image = F.hflip(image)
            label = F.hflip(label)
            return {'image': image, 'label': label}
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'

class RandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, sample):
        if random.random() < self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            image = F.vflip(image)
            label = F.vflip(label)
            return {'image': image, 'label': label}

class RandomRotate(nn.Module):
    def __init__(self, degrees,
                        interpolation=F.InterpolationMode.NEAREST,
                        fill=0,
                        p=0.5):
        super().__init__()
        if not isinstance(degrees, (tuple, list)):
            self.degrees = tuple(-degrees, degrees)
        else:
            self.degrees = degrees
        self.interpolation = interpolation
        self.fill = fill
        self.p = p
    
    def forward(self, sample):
        if random.random() < self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            rotate_degree = random.randrange(self.degrees[0], self.degrees[1])
            image = F.rotate(image, rotate_degree, 
                             interpolation=self.interpolation,
                             fill=self.fill)
            label = F.rotate(label, rotate_degree,
                             interpolation=self.interpolation,
                             fill=self.fill)
            return {'image': image, 'label':label}

class RandomJitter(nn.Module):
    def __init__(self, brightness, contrast, sharpness,
                 p=0.5):
        super().__init__()
        self.p = p
        if not isinstance(brightness, (tuple, list)):
            self.brightness = tuple(1-brightness, 1+brightness)
        else:
            self.brightness = brightness
        
        if not isinstance(contrast, (tuple, list)):
            self.contrast = tuple(1-contrast, 1+contrast)
        else:
            self.contrast = contrast  

        self.sharpness = sharpness
        self.p = p
    
    def forward(self, sample):
        if random.random() < self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            image = F.adjust_brightness(image, brightness_factor)

            if random.random() > self.p:
                contrast_factor = random.uniform(self.contrast[0],self.contrast[1])
                image = F.adjust_contrast(image, contrast_factor)
            
            if random.random() > self.p:
                image = F.adjust_sharpness(image, self.sharpness)
            
            return {'image':image, 'label':label}

class RandomAffine(nn.Module):
    def __init__(self, angle, translate, scale,
                 interpolation=F.InterpolationMode.NEAREST,
                 fill=0,
                 p=0.5):
        super().__init__()
        self.angle = angle # rotate in [-angle, angle]
        self.translate = translate # sequence of integers
        self.scales = scale
        self.p = p
        self.interpolation = interpolation
        self.fill = fill

    def forward(self, sample):
        if random.random() < self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            angle = random.randrange(-self.angle, self.angle)
            scale = random.uniform(self.scales[0], self.scales[1])
            image = F.affine(image, angle, self.translate, scale,
                             shear=0,
                             interpolation=self.interpolation,
                             fill=self.fill)
            label = F.affine(label, angle, self.translate, scale,
                             shear=0,
                             interpolation=self.interpolation,
                             fill=self.fill)
            return {'image': image, 'label': label}

class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, 
                 size,
                 scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=F.InterpolationMode.NEAREST,
        antialias: Optional[Union[str, bool]] = "warn",):
        super().__init__(size)
        self.size = size
        self.interpolation =  interpolation
        self.scale = scale
        self.ratio = ratio
        self.antialias = antialias
    
    def forward(self, sample):
        img, label = sample['image'], sample['label']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        label = F.resized_crop(label, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        return {'image': img, 'label': label}

def collate_fn(samples):
    images = [sample['image'] for sample in samples]
    labels = [sample['label'] for sample in samples]

    images = torch.stack(images)
    labels = torch.stack(labels)

    if labels.ndim == 4: # remove channel dim
        labels = labels.squeeze(1)
    
    return images, labels

def transform(type='train', size=224):
    if type == 'train':
        return transforms.Compose([ ToPILImage(), # most transforms work only for PILImage/tensor
                                    Resize((size, size)),
                                    # RandomRotate((0,360)),
                                    RandomAffine(angle=180, translate=[0,0.2], scale=[0.8,1.2]),
                                    RandomJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), sharpness=5.0),
                                    RandomHorizontalFlip(),
                                    RandomVerticalFlip(),
                                    ToTensor(), # it will automatically scale tensor to [0, 1]
                                    Normalize((0.1365,),(0.1661,))
                                ])
    else:
        return transforms.Compose([ ToPILImage(),
                                    Resize((size, size)),
                                    ToTensor(),
                                    Normalize((0.1365,),(0.1661,))
                                ])

def dataloader(base_dir='./data',
                split='train',
               batch_size=16,
               collate_fn=collate_fn,
               transform=transform,
               size=224,
               **kwargs):
    
    tf = transform(split, size)
    datasets = ACDCDatasets(base_dir, split=split,transform=tf)
    return DataLoader(datasets, 
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    drop_last=True,
                    **kwargs)

def distributed_dataloader(base_dir='./data',
                split='train',
               batch_size=16,
               collate_fn=collate_fn,
               transform=transform,
               size=224,
               **kwargs):
    tf = transform(split, size)
    datasets = ACDCDatasets(base_dir, split=split, transform=tf)
    data_sampler = distributed.DistributedSampler(datasets, shuffle=True)
    return DataLoader(datasets,
                    batch_size=batch_size,
                    shuffle=(data_sampler is None),
                    collate_fn=collate_fn,
                    sampler=data_sampler,
                    drop_last=True)

if __name__ == '__main__':

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/slices')
    args = parser.parse_args()
    trainloader = dataloader(args.data, shuffle=True)

    img, label = iter(trainloader).__next__()

    import matplotlib.pyplot as plt

    print(torch.unique(label[0]))

    _, axes = plt.subplots(2,1)
    axes[0].imshow(img[0].squeeze().numpy(), cmap='gray')
    mask = np.ma.masked_where(label[0].numpy() < 0.5, label[0].numpy())
    axes[0].imshow(mask, alpha=0.5, cmap='jet')
    axes[1].hist(label[0].numpy().ravel())
    plt.show()

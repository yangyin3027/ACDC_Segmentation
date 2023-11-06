import numpy as np
import random
from scipy import ndimage as ndim
from PIL import Image

import torch
from torchvision.transforms import functional as F

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndim.rotate(image, angle, order=3, reshape=False)
    label = ndim.rotate(label, angle, order=0, reshape=False)
    return image, label

def to_pil_image(image, label):
    # uin8 image for further imaging manipulation,
    # such as RadnomJitter, RandomAffine, they work on PIL image
    image = (image*255.).astype(np.uint8)

    image = F.to_pil_image(image)
    # RandomJitter won't be applied to label
    # label should be float to convert to pil_image
    label = F.to_pil_image(label)
    return image, label

def to_tensor(image, label):
    # uint8 type will be squeezed to [0, 1]
    # it will automatically add channel to [c, h, w] form
    # squeeze channel info on label
    image = F.to_tensor(image)
    label = F.to_tensor(label)
    label = label.squeeze(0)

    return image, label

def random_jitter(image):
    if not isinstance(image, (Image.Image, torch.Tensor)):
        raise TypeError('only PIL.Image or torch.Tensor is supported')

    if random.random() > 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        image = F.adjust_brightness(image, brightness_factor)
    if random.random() > 0.5:
        contrast_factor = random.uniform(0.8, 1.2)
        image = F.adjust_contrast(image, contrast_factor)
    if random.random() > 0.5:
        sharpness_factor = random.randrange(0, 6)
        image = F.adjust_sharpness(image, float(sharpness_factor))
    return image

def random_affine(image, label):
    if random.random() > 0.5:
        angle = random.randrange(-180, 180)
        scale = random.uniform(0.8, 1.2)
        translate = [0, 0.2]

        image = F.affine(image, angle, translate, scale,
                         shear=0,
                         interpolation=F.InterpolationMode.NEAREST,
                         fill=0)
        label = F.affine(label, angle, translate, scale,
                         shear=0,
                         interpolation=F.InterpolationMode.NEAREST,
                         fill=0)
    return image, label

class RandomAugment(object):
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, sample, type='train'):
        image, label = sample['image'], sample['label']

        if type =='train':
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            
            if random.random() > 0.5:
                image, label = random_rotate(image, label)
        
        w, h = image.shape
        if w != self.output_size[0] or h != self.output_size[1]:
            scale = (self.output_size[0]/w, self.output_size[1]/h)
            image = ndim.zoom(image, scale, order=3)
            label = ndim.zoom(label, scale, order=0)
        
        # convert to PIL.Image for further augmentation
        image, label = to_pil_image(image, label)

        if type =='train':
            # apply random jitter
            if random.random() > 0.5:
               image = random_jitter(image)
            # apply random affine
            if random.random() > 0.5:
                image, label = random_affine(image, label)
        # convert to tensor
        image, label = to_tensor(image, label)

        sample = {'image': image, 'label': label}
        return sample
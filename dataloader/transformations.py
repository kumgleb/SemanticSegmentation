import numpy as np
import cv2
import albumentations as albu
from skimage.transform import resize
from typing import Dict


class TargetMask:
    def __init__(self,
                 labels_range: Dict,
                 erosion_kernel: tuple,
                 dilation_kernel: tuple):
        self.labels_range = labels_range
        self.erosion_kernel = erosion_kernel
        self.dilation_kernel = dilation_kernel

    def create_mask(self, y):
        h, w, _ = y.shape
        mask = np.zeros((h, w))
        for c, label in enumerate(self.labels_range.keys()):
            idxs = np.where(np.logical_and(
                np.all(y <= self.labels_range[label][1], axis=-1),
                np.all(y >= self.labels_range[label][0], axis=-1))
            )
            mask[idxs[1], idxs[0]] = c + 1

        if self.erosion_kernel:
            kernel = np.ones(self.erosion_kernel, dtype='uint8')
            mask = cv2.erode(mask, kernel, iterations=1)  
        if self.dilation_kernel:
            kernel = np.ones(self.dilation_kernel, dtype='uint8')
            mask = cv2.dilate(mask, kernel, iterations=1)     

        return mask

    def __call__(self, x, y):
        y = self.create_mask(y)
        return x, y


class Normalize:
    def __init__(self):
        self.norm = albu.Normalize()

    def __call__(self, x, y):
        out = self.norm(image=x)
        x = out['image']
        return x, y


class Resize:
    def __init__(self,
                 out_size: tuple):
        self.out_size = out_size

    def __call__(self, x, y):
        x = resize(x, self.out_size)
        y = resize(y, self.out_size)
        return x, y


class Augmentations:
    def __init__(self,
                 p_flip: float):
        self.flip = albu.Flip(p=p_flip)

    def __call__(self, x, y):
        aug_dict = self.flip(image=x, mask=y)
        x = aug_dict['image']
        y = aug_dict['mask']
        return x, y


class MoveAxis:
    def __init__(self):
        pass

    def __call__(self, x, y):
        x = np.moveaxis(x, -1, 0)
        y = np.moveaxis(y, -1, 0)
        return x, y


class Compose:
    def __init__(self,
                 transforms: list):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y

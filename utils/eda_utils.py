import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage


def plot_samples(path):
    samples = os.listdir(path)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax = ax.ravel()

    for s in range(len(ax)):
        random_sample = np.random.choice(samples)
        path_to_sample = os.path.join(path, random_sample)
        sample = Image.open(path_to_sample)
        ax[s].imshow(sample)
    fig.tight_layout()


def get_sample(path):
    samples = os.listdir(path)
    random_sample = np.random.choice(samples)
    path_to_sample = os.path.join(path, random_sample)
    sample = np.asanyarray(Image.open(path_to_sample))
    return sample


def probe_colors(sample, points):
    fig = plt.figure(figsize=(18, 12))
    plt.imshow(sample)
    for point in points:
        plt.scatter(point[1], point[0], s=3, c='w')
        plt.text(point[1], point[0],
                 str(sample[point[0], point[1], :]),
                 color='w',
                 fontsize=14)


def calc_grad(img):
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    x = ndimage.convolve(img, kx)
    y = ndimage.convolve(img, ky)
    grad = np.hypot(x, y)
    return grad


def plot_grads(sample):
    fig, ax = plt.subplots(1, 3, figsize=(14, 6))
    mask = sample[:, 256:, :]
    for i, c in enumerate(['R', 'G', 'B']):
        grad = calc_grad(mask[:, :, i]).astype('uint8')
        ax[i].imshow(grad)
        ax[i].set_title(f'Grad of {c} channel')


def create_mask(sample, labels_range, erode_dilate=False):
    h, w, _ = sample.shape
    mask = np.zeros((h, w))

    for c, label in enumerate(labels_range.keys()):
        idxs = np.where(np.logical_and(
            np.all(sample <= labels_range[label][1], axis=-1),
            np.all(sample >= labels_range[label][0], axis=-1))
        )
        mask[idxs[0], idxs[1]] = c + 1

    if erode_dilate:
        kernel = np.ones((2, 2), dtype='uint8')
        mask = cv2.erode(mask, kernel, iterations=1)
        kernel = np.ones((3, 3), dtype='uint8')
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def test_masking(path, labels_range):
    fig, ax = plt.subplots(3, 3, figsize=(16, 8), sharey=True)
    for i in range(3):
        sample = get_sample(path)
        tgt = sample[:, 256:, :]
        mask = create_mask(tgt, labels_range, erode_dilate=False)
        mask_d = create_mask(tgt, labels_range, erode_dilate=True)

        ax[i][0].imshow(tgt)
        ax[i][1].imshow(mask)
        ax[i][2].imshow(mask_d)

    ax[0][0].set_title('Original')
    ax[0][1].set_title('From range')
    ax[0][2].set_title('With erode-dilate')


def x_to_img(x):
    x = x.permute(1, 2, 0).cpu().numpy() 
    x = (x * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    x = (x*255).astype(np.int64)
    return x

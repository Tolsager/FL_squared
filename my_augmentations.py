from typing import Union
from PIL import Image
import torchvision.transforms.functional as F

import torchvision
import matplotlib.pyplot as plt

cifar = torchvision.datasets.CIFAR10(root="data/raw")

image = cifar[-57][0]


def plot_rotation():
    fig, ax = plt.subplots(1, 2)
    im1 = F.rotate(image, -10)
    im2 = F.rotate(image, 10)

    texts = ["-10 degrees", "+10 degrees"]
    images = [im1, im2]
    plt.suptitle("Rotation", fontsize=20, y=0.9)
    for im, a, t in zip(images, ax, texts):
        a.axis("off")
        a.imshow(im)
        a.set_title(t)
    plt.show()


def plot_crop():
    crop_transform = torchvision.transforms.RandomCrop(32, 4)
    fig, ax = plt.subplots(1, 2)
    im1 = crop_transform(image)
    im2 = crop_transform(image)

    images = [im1, im2]
    plt.suptitle("Random crop", fontsize=20, y=0.9)
    for im, a in zip(images, ax):
        a.axis("off")
        a.imshow(im)
    plt.show()


def plot_binary():
    fig, ax = plt.subplots(1, 2)
    im1 = F.hflip(image)
    im2 = F.to_grayscale(image)
    images = [im1, im2]
    plt.suptitle("Binary augmentations", fontsize=20, y=0.9)
    texts = ["Horizontal flip", "Grayscale"]
    for im, a, t in zip(images, ax, texts):
        a.axis("off")
        if t == "Grayscale":
            a.imshow(im, cmap="gray")
        else:
            a.imshow(im)
        a.set_title(t)
    plt.show()


# plot_rotation()
# plot_crop()
plot_binary()

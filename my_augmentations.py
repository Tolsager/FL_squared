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

def plot_resized_crop():
    fig, ax = plt.subplots(1, 2)
    transform = torchvision.transforms.RandomResizedCrop(23, (0.2,1))
    im1 = transform(image)
    im2 = transform(image)
    images = [im1, im2]
    plt.suptitle("Random resized crop", fontsize=20, y=0.9)
    for im, a in zip(images, ax):
        a.axis("off")
        a.imshow(im)
    plt.show()

def plot_brightness():
    fig, ax = plt.subplots(1, 2)
    brightness = 0.4
    im1 = F.adjust_brightness(image, 1-brightness)
    im2 = F.adjust_brightness(image, 1+brightness)

    images = [im1, im2]
    texts = ["Brightness factor: 0.4", "Brightness factor: 1.6"]
    plt.suptitle("Brightness", fontsize=20, y=0.9)
    for im, a, t in zip(images, ax, texts):
        a.axis("off")
        a.imshow(im)
        a.set_title(t)
    plt.show()

def plot_contrast():
    fig, ax = plt.subplots(1, 2)
    brightness = 0.4
    im1 = F.adjust_contrast(image, 1-brightness)
    im2 = F.adjust_contrast(image, 1+brightness)

    images = [im1, im2]
    texts = ["Contrast factor: 0.4", "Contrast factor: 1.6"]
    plt.suptitle("Contrast", fontsize=20, y=0.9)
    for im, a, t in zip(images, ax, texts):
        a.axis("off")
        a.imshow(im)
        a.set_title(t)
    plt.show()

def plot_saturation():
    fig, ax = plt.subplots(1, 2)
    brightness = 0.4
    im1 = F.adjust_saturation(image, 1-brightness)
    im2 = F.adjust_saturation(image, 1+brightness)

    images = [im1, im2]
    texts = ["Saturation factor: 0.4", "Saturation factor: 1.6"]
    plt.suptitle("Saturation", fontsize=20, y=0.9)
    for im, a, t in zip(images, ax, texts):
        a.axis("off")
        a.imshow(im)
        a.set_title(t)
    plt.show()

def plot_hue():
    fig, ax = plt.subplots(1, 2)
    im1 = F.adjust_hue(image, -0.1)
    im2 = F.adjust_hue(image, 0.1)

    images = [im1, im2]
    texts = ["Hue factor: -0.1", "Hue factor: 0.1"]
    plt.suptitle("Hue", fontsize=20, y=0.9)
    for im, a, t in zip(images, ax, texts):
        a.axis("off")
        a.imshow(im)
        a.set_title(t)
    plt.show()

def plot_combined():
    fig, ax = plt.subplots(1, 4)
    augmentations = [
        torchvision.transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomApply(
            [
                torchvision.transforms.ColorJitter(
                    0.4, 0.4, 0.4, 0.1
                )  # not strengthened
            ],
            p=0.8,
        ),
        torchvision.transforms.RandomGrayscale(p=0.2),
        # torchvision.transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    ]
    transform = torchvision.transforms.Compose(augmentations)
    images = [transform(image) for _ in range(4)]
    plt.suptitle("SimSiam augmentations", fontsize=20, y=0.7)
    for im, a in zip(images, ax):
        a.axis("off")
        a.imshow(im)

    plt.show()


def plot_colorjitter():
    fig, ax = plt.subplots(1, 4)
    augmentations = [
                torchvision.transforms.ColorJitter(
                    0.4, 0.4, 0.4, 0.1
                )  # not strengthened
            ]
    transform = torchvision.transforms.Compose(augmentations)
    images = [transform(image) for _ in range(4)]
    plt.suptitle("Color jitter", fontsize=20, y=0.7)
    for im, a in zip(images, ax):
        a.axis("off")
        a.imshow(im)

    plt.show()


# plot_rotation()
# plot_crop()
# plot_binary()
# plot_resized_crop()
# plot_brightness()
# plot_contrast()
# plot_saturation()
# plot_hue()
# plot_combined()
plot_colorjitter()


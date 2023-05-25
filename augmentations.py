from typing import Union
from PIL import Image

import torchvision
import matplotlib.pyplot as plt

cifar = torchvision.datasets.CIFAR10(root="data/raw")

image = cifar[-57][0]


def plot_augmentations(image: Image, transforms: list):
    fig, ax = plt.subplots(len(transforms) + 1, 5)

    ax[0, 0].set_title("Original Image", fontsize=8)

    for i, transform in enumerate(transforms):
        ax[i, 0].imshow(image)
        ax[i, 0].set_xticklabels([])
        ax[i, 0].set_yticklabels([])
        ax[i, 0].tick_params(axis="both", which="both", length=0)
        ax[i, 0].set_ylabel(type(transform).__name__ if type(transform).__name__ != "RandomApply" else "ColorJitter", fontsize=5)
        for j in range(1, 5):
            ax[i, j].axis("off")
            ax[i, j].imshow(transform(image))

    ax[-1, 0].imshow(image)
    ax[-1, 0].set_xticklabels([])
    ax[-1, 0].set_yticklabels([])
    ax[-1, 0].tick_params(axis="both", which="both", length=0)
    for j in range(1, 5):
        ax[-1, j].axis("off")
        ax[-1, j].imshow(torchvision.transforms.Compose(transforms)(image))
    ax[-1, 0].set_ylabel("Combined", fontsize=5)

    plt.subplots_adjust(left=0.04,
                        bottom=0.03,
                        right=0.750,
                        top=0.85,
                        wspace=0.05,
                        hspace=0.2)

    # Use to adjust the plot interactively
    # plt.subplot_tool()


    plt.savefig(f"1.png", bbox_inches="tight", dpi=300)
    plt.show()

crop_transform = torchvision.transforms.RandomCrop(32, 4)
rotation_transform = torchvision.transforms.RandomRotation(10)
flip_transform = torchvision.transforms.RandomHorizontalFlip(0.5)
colorjitter = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
resized_crop = torchvision.transforms.RandomResizedCrop(32, scale=(0.2, 1.0))
grayscale = torchvision.transforms.RandomGrayscale(p=0.4)

simple_transforms = [crop_transform, rotation_transform, flip_transform]
simsiam_transforms = [resized_crop, flip_transform, torchvision.transforms.RandomApply([colorjitter], p=0.8), grayscale]


plot_augmentations(image, simple_transforms)


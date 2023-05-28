import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Union


def plot_simsiam_images(dataset, index):
    im1, im2, label = dataset[index]
    plot_two_images_side_by_side(im1, im2)


def plot_two_images_side_by_side(im1, im2):
    fig, axs = plt.subplots(1, 2)
    im1 = im1.permute(1, 2, 0).numpy()
    im2 = im2.permute(1, 2, 0).numpy()
    axs[0].imshow(im1, vmin=im1.min(), vmax=im1.max())
    axs[1].imshow(im2, vmin=im2.min(), vmax=im2.max())
    plt.show()


def frac_grad(model: torch.nn.Module) -> float:
    n_params = 0
    n_tracked = 0
    for param in model.parameters():
        n_params += 1
        if param.requires_grad:
            n_tracked += 1
    return n_tracked / n_params


def are_models_identical(m1: torch.nn.Module, m2: torch.nn.Module):
    """Tests if the state dicts of the two models are identical
    Expects both models to be on the same device and have the same
    architecture

    Args:
        m1 (torch.nn.Module): model1
        m2 (torch.nn.Module): model2
    """

    for v1, v2 in zip(m1.state_dict().values(), m2.state_dict().values()):
        if not torch.allclose(v1, v2):
            return False
    return True


def get_dataset_distribution(datasets: list[Union[torch.utils.data.Dataset, torch.utils.data.Subset]]):
    distributions = []
    for dataset in datasets:
        labels = []
        for *_, label in dataset:
            labels.append(label)
        distributions.append(np.bincount(labels))
    return distributions


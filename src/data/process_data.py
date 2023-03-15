import random
from collections.abc import Callable, Iterable
from typing import Any, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import ImageFilter

cifar10_standard_transforms = [
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
]

imagenet_standard_transforms = [
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
]


def shuffle_dataset(dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    indices = np.random.choice(len(dataset), len(dataset), replace=False)
    return torch.utils.data.Subset(dataset, indices)


class DataSplitter:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_clients: int,
        shards_per_client: int = 2,
    ):
        self.dataset = dataset
        self.n_clients = n_clients
        self.shards_per_client = shards_per_client
        self.n_shards = shards_per_client * n_clients
        self.n_samples = len(self.dataset)

    def get_shard_start_indices(self) -> np.ndarray:
        """generates the start indices of each shard

        Returns:
            np.ndarray: array of indices
        """
        return np.linspace(0, self.n_samples, num=self.n_shards, endpoint=False).astype(
            int
        )

    def split_data(self) -> list[torch.utils.data.Dataset]:
        """uses the shard start indices to get all the shards
        from the data and combine them into the client datasets

        Returns:
            list[torch.utils.data.Dataset]: client datasets
        """
        shard_indices = self.get_shard_start_indices()

        # randomly select shards_per_client shards
        shard_assignments = np.random.choice(
            range(len(shard_indices)),
            size=(self.n_clients, self.shards_per_client),
            replace=False,
        )

        # append extra element to shard_indices to prevent going out of bounds when
        # splitting data
        shard_indices = np.append(shard_indices, self.n_samples)

        # from the shard indices assigned to each client, get the
        # shards as pytorch subsets
        client_datasets = []
        for split in shard_assignments:
            temp_datasets = []
            for shard_idx in split:
                start_idx = shard_indices[shard_idx]
                end_idx = shard_indices[shard_idx + 1]
                data_subset = torch.utils.data.Subset(
                    self.dataset, range(start_idx, end_idx)
                )
                temp_datasets.append(data_subset)

            client_dataset = torch.utils.data.ConcatDataset(temp_datasets)
            client_datasets.append(client_dataset)

        return client_datasets


def sort_torch_dataset(
    dataset: torch.utils.data.Dataset, sort_fn: Callable[[Iterable], Any]
) -> torch.utils.data.Dataset:
    # sort the data by label
    sorted_ds = sorted(dataset, key=sort_fn)
    return sorted_ds


def cifar10_sort_fn(batch: tuple[torch.Tensor]):
    # return the label
    return batch[1]


def train_val_split(
    dataset: torch.utils.data.Dataset,
    val_size: Union[int, float],
    shuffle: bool = False,
) -> Tuple[torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset]:

    if shuffle:
        indices = np.random.choice(len(dataset), len(dataset), replace=False)
    else:
        indices = range(len(dataset))

    if isinstance(val_size, int):
        val_split = torch.utils.data.Subset(dataset, indices[:val_size])
        test_split = torch.utils.data.Subset(dataset, indices[val_size:])
    else:
        # calculate number of samples in the validation split
        n_val_samples = int(val_size * len(dataset))
        val_split = torch.utils.data.Subset(dataset, indices[:n_val_samples])
        test_split = torch.utils.data.Subset(dataset, indices[n_val_samples:])
    return val_split, test_split


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transforms: torchvision.transforms.transforms.Compose,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        augmented_image = self.transforms(image)
        return augmented_image, label


class SimSiamDataset(AugmentedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transforms: torchvision.transforms.transforms.Compose,
    ):
        super().__init__(dataset, transforms)
        self.standard_transforms = torchvision.transforms.Compose(
            cifar10_standard_transforms
        )

    def __getitem__(self, i):
        image, label = self.dataset[i]
        aug1 = self.transforms(image)
        aug2 = self.transforms(image)
        image = self.standard_transforms(image)
        return aug1, aug2, image, label


def get_cifar10_transforms() -> torchvision.transforms.transforms.Compose:
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            # Flips the image w.r.t horizontal axis
            torchvision.transforms.RandomRotation(10),
        ].extend(cifar10_standard_transforms)
    )
    return transforms


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_simsiam_transforms(
    img_size: Union[tuple[int, int], int]
) -> torchvision.transforms.transforms.Compose:
    augmentations = [
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        torchvision.transforms.RandomApply(
            [
                torchvision.transforms.ColorJitter(
                    0.4, 0.4, 0.4, 0.1
                )  # not strengthened
            ],
            p=0.8,
        ),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
    ]
    augmentations.extend(cifar10_standard_transforms)
    return torchvision.transforms.transforms.Compose(augmentations)

import random
from collections.abc import Callable, Iterable
from typing import Any, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import ImageFilter

CIFAR10_STANDARD_TRANSFORMS = [
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    torchvision.transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
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


def sort_dataset(
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
    """splits a dataset into two splits

    Args:
        dataset (torch.utils.data.Dataset): dataset to split
        val_size (Union[int, float]): number of samples or fraction
          of samples to use for the val split
        shuffle (bool, optional): shuffles the data. Defaults to False.

    Raises:
        ValueError: validation size is larger than number of samples

    Returns:
        Tuple[torch.utils.data.dataset.Subset,
          torch.utils.data.dataset.Subset]: the splits
    """
    if shuffle:
        indices = np.random.choice(len(dataset), len(dataset), replace=False)
    else:
        indices = range(len(dataset))

    if isinstance(val_size, int):
        if val_size > len(dataset):
            raise ValueError(
                "The validation size is larger than the samples in the dataset"
            )
        train_split = torch.utils.data.Subset(dataset, indices[val_size:])
        val_split = torch.utils.data.Subset(dataset, indices[:val_size])
    else:
        # calculate number of samples in the validation split
        n_val_samples = int(val_size * len(dataset))
        if n_val_samples > len(dataset):
            raise ValueError(
                "The validation size is larger than the samples in the dataset"
            )
        train_split = torch.utils.data.Subset(dataset, indices[n_val_samples:])
        val_split = torch.utils.data.Subset(dataset, indices[:n_val_samples])
    return train_split, val_split


def stratified_train_val_split(
    dataset: torch.utils.data.Dataset, label_fn: Callable, val_size: float
) -> Tuple[torch.utils.data.Dataset]:
    """sorts the data and returns two stratified datasets

    Expects each class to have more than one sample

    Args:
        dataset (torch.utils.data.Dataset): dataset to split
        label_fn (Callable): function that takes a dataset sample as
          input and returns the label
        val_size (float): fraction of samples for validation

    Returns:
        Tuple(torch.utils.data.Dataset): stratified datasets
    """
    # calculate number of samples in the validation split
    n_val_samples = int(val_size * len(dataset))
    if n_val_samples > len(dataset):
        raise ValueError(
            "The validation size is larger than the samples in the dataset"
        )

    sorted_ds = sort_dataset(dataset, sort_fn=label_fn)
    label_start_end_index = {}

    last_label = None
    for i, sample in enumerate(sorted_ds):
        label = label_fn(sample)
        if i == 0:
            label_start_end_index[label] = [0]
        elif label not in label_start_end_index:
            label_start_end_index[last_label].append(i - 1)
            label_start_end_index[label] = [i]
        last_label = label

    label_start_end_index[last_label].append(i)

    all_train_indices = []
    all_val_indices = []
    for indices in label_start_end_index.values():
        # number of samples with a specific label
        n_samples = indices[1] - indices[0] + 1

        n_val_samples = int(n_samples * val_size)
        val_indices = list(range(indices[0], indices[0] + n_val_samples))
        train_indices = list(range(indices[0] + n_val_samples, indices[1] + 1))

        all_val_indices.extend(val_indices)
        all_train_indices.extend(train_indices)

    val_ds = torch.utils.data.Subset(dataset, all_val_indices)
    train_ds = torch.utils.data.Subset(dataset, all_train_indices)

    return train_ds, val_ds


class AugmentedDataset(torch.utils.data.Dataset):
    """applies a transformation to a dataset

    Args:
        dataset: dataset to augment
        transforms: transforms to apply to the dataset
    """

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
    """applies the augmentations to create two augmented
      images. Also returns the unaugmented image but scaled

    Args:
        dataset: dataset to augment
        transforms: transforms to apply to the images used for training
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transforms: torchvision.transforms.transforms.Compose,
    ):
        super().__init__(dataset, transforms)
        self.standard_transforms = torchvision.transforms.Compose(
            CIFAR10_STANDARD_TRANSFORMS
        )

    def __getitem__(self, i):
        image, label = self.dataset[i]
        aug1 = self.transforms(image)
        aug2 = self.transforms(image)
        image = self.standard_transforms(image)
        return aug1, aug2, image, label


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_simsiam_transforms(
    img_size: Union[tuple[int, int], int] = 32, min_scale: float = 0.2
) -> torchvision.transforms.transforms.Compose:
    augmentations = [
        torchvision.transforms.RandomResizedCrop(img_size, scale=(min_scale, 1.0)),
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
    augmentations.extend(CIFAR10_STANDARD_TRANSFORMS)
    return torchvision.transforms.Compose(augmentations)


def simple_datasplit(
    dataset: torch.utils.data.Dataset, n_splits: int
) -> list[torch.utils.data.Dataset]:
    """splits the dataset into n_splits of the same size by
    assigning the first sample to the first dataset, second sample
    to the second dataset etc.

    Args:
        dataset (torch.utils.data.Dataset): dataset to split
        n_splits (int): number of splits
    """

    datasets_idices = [[] for i in range(n_splits)]
    n_samples = len(dataset)
    for i in range(n_samples):
        datasets_idices[i % n_splits].append(i)

    datasets = [torch.utils.data.Subset(dataset, i) for i in datasets_idices]
    return datasets

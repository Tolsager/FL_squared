import os

import numpy as np
import pytest
import torch
import torchvision

from src.data.process_data import (
    DataSplitter,
    cifar10_sort_fn,
    sort_dataset,
    stratified_train_val_split,
    train_val_split,
)

save_path = "data/raw"
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(
    root=save_path, train=False, transform=transforms, download=True
)
dataset = torch.utils.data.Subset(dataset, range(100))


class TestDatasplitter:
    n_clients = [5, 10]
    shards_per_client = [2, 1]
    datasplitters = [
        DataSplitter(dataset, clients, shards)
        for clients, shards in zip(n_clients, shards_per_client)
    ]
    dataset = dataset

    def test_get_shard_indices(self):
        for datasplitter in self.datasplitters:
            shard_indices = datasplitter.get_shard_start_indices()
            assert np.all(
                shard_indices == np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
            )

    def test_split_data(self):
        for i, datasplitter in enumerate(self.datasplitters):
            splits = datasplitter.split_data()
            assert len(splits) == self.n_clients[i]
            len_split0 = len(splits[0])

            for split in splits:
                assert isinstance(split, torch.utils.data.Dataset)
                assert len(split) >= len_split0 and len(split) < (len_split0 + 2)


def test_sort_torch_dataset():
    sort_fn = cifar10_sort_fn
    sorted_ds = sort_dataset(dataset, sort_fn)
    prev = sorted_ds[0]
    # test that the data is strictly increasing which implies that it's sorted
    for i in range(1, len(sorted_ds)):
        assert sorted_ds[i][1] >= sorted_ds[i - 1][1]


def test_train_val_split():
    dataset = torchvision.datasets.CIFAR10(
        root=save_path, train=False, transform=transforms, download=True
    )
    ds_train, ds_val = train_val_split(dataset, 60, shuffle=True)

    assert isinstance(ds_train, torch.utils.data.dataset.Subset)
    assert isinstance(ds_val, torch.utils.data.dataset.Subset)

    assert len(ds_train) == 60
    assert len(ds_val) == len(dataset) - 60


def test_stratified_train_val_split():
    dataset = torchvision.datasets.CIFAR10(
        root=save_path, train=False, transform=transforms, download=True
    )
    ds_train, ds_val = stratified_train_val_split(
        dataset, val_size=0.5, label_fn=lambda x: x[1]
    )

    assert isinstance(ds_train, torch.utils.data.dataset.Subset)
    assert isinstance(ds_val, torch.utils.data.dataset.Subset)

    assert len(ds_train) - 1 < len(ds_val) < len(ds_train) + 1

    label_diffs = {}
    for (_, label_train), (_, label_val) in zip(ds_train, ds_val):
        label_diffs[label_train] = label_diffs.get(label_train, 0) + 1
        label_diffs[label_val] = label_diffs.get(label_val, 0) - 1

    for v in label_diffs.values():
        assert abs(v) < 2

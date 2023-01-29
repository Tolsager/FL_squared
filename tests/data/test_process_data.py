import torch
import torchvision
import numpy as np
from src.data.process_data import DataSplitter, sort_torch_dataset, cifar10_sort_fn
import pytest
import os

save_path = "data/raw"
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(
    root=save_path, train=False, transform=transforms, download=False
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
    sorted_ds = sort_torch_dataset(dataset, sort_fn)
    prev = sorted_ds[0]
    # test that the data is strictly increasing which implies that it's sorted
    for i in range(1, len(sorted_ds)):
        assert sorted_ds[i][1] >= sorted_ds[i - 1][1]

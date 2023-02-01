from collections.abc import Callable, Iterable
from typing import Any, Union

import numpy as np
import torch


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


def val_test_split(
    dataset: torch.utils.data.Dataset, val_size: Union[int, float]
) -> list[torch.utils.data.dataset.Subset]:
    if isinstance(val_size, int):
        val_split = torch.utils.data.Subset(dataset, range(val_size))
        test_split = torch.utils.data.Subset(dataset, range(val_size, len(dataset)))
    else:

        # calculate number of samples in the validation split
        n_val_samples = val_size * len(dataset)
        val_split = torch.utils.data.Subset(dataset, range(n_val_samples))
        test_split = torch.utils.data.Subset(
            dataset, range(n_val_samples, len(dataset))
        )
    return [val_split, test_split]

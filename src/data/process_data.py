import torch
import numpy as np


def process_cifar10() -> dict[str : torch.Tensor]:
    # load train data
    pass


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

    def get_shard_indices(self):
        return np.linspace(0, self.n_samples, num=self.n_shards, endpoint=False).astype(
            int
        )


def sort_torch_dataset(dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    # sort the data by label
    sort_fn = lambda x: x[1]
    sorted_ds = sorted(dataset, key=sort_fn)
    return sorted_ds


def split_data_to_clients(
    dataset: torch.utils.data.Dataset, n_clients: int, shards_per_client: int = 2
):
    if not isinstance(dataset, torch.utils.data.Dataset):
        raise ValueError("Expected a torch dataset")

    n_samples = len(dataset)
    total_shards = n_clients * shards_per_client
    shard_size = np.floor(n_samples / total_shards)

    # sort dataset

    # if more shards are requested than there are data point, raise error
    if shard_size == 0:
        raise ValueError(
            "The number of shards needed exceed the number of samples in the dataset provided"
        )

    # create a sampler
    sampler = torch.utils.data.SequentialSampler(dataset)

    #
    batch_sample = torch.utils.data.BatchSampler(
        sampler, batch_size=shard_size, drop_last=False
    )

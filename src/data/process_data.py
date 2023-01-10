import torch
import numpy as np


def process_cifar10() -> dict[str: torch.Tensor]:
    # load train data
    pass

def sort_torch_dataset(dataset: torch.utils.Dataset) -> torch.utils.Dataset:
    # sort the data by label
    sort_fn = lambda x: x[1]
    sorted_ds = sorted(dataset, key=sort_fn)
    return sorted_ds

def split_data_to_clients(dataset: torch.utils.Dataset, n_clients: int, shards_per_client: int = 2):
    if not isinstance(dataset, torch.utils.Dataset):
        raise ValueError("Expected a torch dataset")
    
    n_samples = len(dataset)
    total_shards = n_clients * shards_per_client
    shard_size = np.floor(n_samples / total_shards)

    # sort dataset
    
    # if more shards are requested than there are data point, raise error
    if shard_size == 0:
        raise ValueError("The number of shards needed exceed the number of samples in the dataset provided")
    
    # create a sampler
    sampler = torch.utils.data.SequentialSampler(dataset)
    
    # 
    batch_sample = torch.utils.data.BatchSampler(sampler, batch_size = shard_size, drop_last = False)


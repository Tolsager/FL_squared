import torch
import numpy as np
from src.data.process_data import DataSplitter
import pytest
import os


@pytest.mark.skipif(not os.path.exists("data/raw/test.pt"))
class TestDatasplitter:
    dataset = torch.load("data/raw/test.pt")
    dataset = torch.utils.data.Subset(dataset, 100)
    n_clients = 10
    shards_per_client = 1
    datasplitter = DataSplitter(dataset, n_clients, shards_per_client)

    def test_get_shard_indices(self):
        shard_indices = self.datasplitter.get_shard_indices()
        assert shard_indices == np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

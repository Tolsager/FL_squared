import os

import pytest
import torch

from src.data import make_dataset


def test_download():
    make_dataset.download_dataset()
    assert os.path.exists("data/raw/train.pt")
    assert os.path.exists("data/raw/test.pt")


@pytest.mark.skipif(
    not os.path.exists("data/raw/train.pt"), reason="Dataset not downloaded"
)
@pytest.mark.skipif(
    not os.path.exists("data/raw/test.pt"), reason="Dataset not downloaded"
)
class TestData:
    def test_length(self):
        assert len(torch.load("data/raw/train.pt")) == 50_000
        assert len(torch.load("data/raw/test.pt")) == 10_000

    def test_shape(self):
        train, test = make_dataset.load_dataset()
        assert train[0][0].shape == torch.Size([3, 32, 32])
        assert test[0][0].shape == torch.Size([3, 32, 32])

    def test_type(self):
        train, test = make_dataset.load_dataset()
        assert train[0][0].dtype == torch.float32
        assert test[0][0].dtype == torch.float32

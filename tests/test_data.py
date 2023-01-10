import pytest
import torch
from src.data import make_dataset
import os


def test_download():
    make_dataset.download_dataset()
    assert os.path.exists("data/raw/train.pt") and os.path.exists("data/raw/test.pt")


def test_length():
    assert len(torch.load("data/raw/train.pt")) == 50_000
    assert len(torch.load("data/raw/test.pt")) == 10_000


def test_shape():
    train, test = make_dataset.load_dataset()
    assert train[0][0].shape == torch.Size([3, 32, 32])
    assert test[0][0].shape == torch.Size([3, 32, 32])


def test_type():
    train, test = make_dataset.load_dataset()
    assert train[0][0].dtype == torch.float32
    assert test[0][0].dtype == torch.float32
import os

import PIL
import pytest
import torch

from src.data import make_dataset


def test_download():
    for dataset in make_dataset.DOWNLOADABLE_DATASETS:
        make_dataset.download_dataset(dataset=dataset)
        assert os.path.exists(f"data/raw/{dataset}/train.pt")
        assert os.path.exists(f"data/raw/{dataset}/test.pt")

# -*- coding: utf-8 -*-
import gzip
import shutil
import logging
import os
from pathlib import Path
from typing import Tuple, Optional

import urllib.request
import click
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv

downloadable_datasets = {"cifar10"}

def download_dataset(save_path: str = "data/raw", dataset: str = "cifar10") -> None:
    """downloads dataset

    Args:
        save_path (str, optional): directory to store datasets in. Defaults to "data/raw".
        dataset (str, optional): name of the dataset. Defaults to "cifar10".

    Raises:
        ValueError: dataset is not implemented
    """
    if dataset not in downloadable_datasets:
        raise ValueError(f"{dataset} is not implemented yet.\nAvailable datasets: {downloadable_datasets}")
    save_dir = os.path.join(save_path, dataset)
    os.makedirs(save_dir, exist_ok=True)

    if dataset == "cifar10":
        train = torchvision.datasets.CIFAR10(root=save_path, train=True, download=True)
        test = torchvision.datasets.CIFAR10(root=save_path, train=False, download=True)

        train = torchvision.datasets.ImageNet(root=save_dir, train=True, download=True)
        test = torchvision.datasets.ImageNet(root=save_dir, train=False, download=True)

    torch.save(train, f"{save_dir}/train.pt")
    torch.save(test, f"{save_dir}/test.pt")


def load_dataset(
    load_path: str = "data/raw", n_samples: Optional[int] = None
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train = torch.load(os.path.join(load_path, "train.pt"))
    test = torch.load(os.path.join(load_path, "test.pt"))

    if n_samples is not None:
        train = torch.utils.data.Subset(train, range(n_samples))
        test = torch.utils.data.Subset(test, range(n_samples))

    return train, test


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    download_dataset(input_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # main()
    download_dataset()

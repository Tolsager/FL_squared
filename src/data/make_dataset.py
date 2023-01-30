# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import Tuple

import click
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv


def download_dataset(save_path: str = "data/raw") -> None:
    """Download the CIFAR10 dataset and apply tensor conversion."""
    # TODO: Determine whether or not to normalize based on entire population,
    # since clients will not have the full set.
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train = torchvision.datasets.CIFAR10(
        root=save_path, train=True, transform=transforms, download=True
    )
    test = torchvision.datasets.CIFAR10(
        root=save_path, train=False, transform=transforms, download=True
    )

    torch.save(train, "data/raw/train.pt")
    torch.save(test, "data/raw/test.pt")


def load_dataset(
    load_path: str = "data/raw", n_samples: int = None
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

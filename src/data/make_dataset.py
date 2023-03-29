# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import click
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv

DOWNLOADABLE_DATASETS = {"cifar10"}


def download_dataset(save_path: str = "data/raw", dataset: str = "cifar10") -> None:
    """downloads dataset

    Args:
        save_path (str, optional): directory to store datasets in. Defaults
            to "data/raw".
        dataset (str, optional): name of the dataset. Defaults to "cifar10".

    Raises:
        ValueError: dataset is not implemented
    """
    if dataset not in DOWNLOADABLE_DATASETS:
        raise ValueError(
            f"{dataset} is not implemented yet.\n\
            Available datasets: {DOWNLOADABLE_DATASETS}"
        )
    save_dir = os.path.join(save_path, dataset)
    os.makedirs(save_dir, exist_ok=True)

    if dataset == "cifar10":
        train = torchvision.datasets.CIFAR10(root=save_path, train=True, download=True)
        test = torchvision.datasets.CIFAR10(root=save_path, train=False, download=True)

    torch.save(train, f"{save_dir}/train.pt")
    torch.save(test, f"{save_dir}/test.pt")


def load_dataset(
    load_path: str = "data/raw",
    n_train_samples: Optional[int] = None,
    n_test_samples: Optional[int] = None,
    dataset: str = "cifar10",
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if dataset not in DOWNLOADABLE_DATASETS:
        raise ValueError(
            f"{dataset} is not implemented yet.\n\
            Available datasets: {DOWNLOADABLE_DATASETS}"
        )
    train_file = os.path.join(load_path, dataset, "train.pt")
    test_file = os.path.join(load_path, dataset, "test.pt")

    train = torch.load(train_file)
    test = torch.load(test_file)

    datasets = [train, test]
    for i, n_samples, ds in enumerate(zip([n_train_samples, n_test_samples], datasets)):
        if n_samples is not None:
            if n_samples > len(ds):
                raise ValueError(
                    "The number of samples requested is larger than the dataset"
                )
            else:
                datasets[i] = torch.utils.data.Subset(train, range(n_samples))

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

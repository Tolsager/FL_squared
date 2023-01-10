# -*- coding: utf-8 -*-
import click
import logging
import torchvision
import torch
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def download_dataset(save_path: str = 'data/raw') -> None:
    """Download the CIFAR10 dataset and apply normalization and tensor conversion."""

    # TODO: Determine whether or not to normalize based on entire population, since clients will not have the full set.
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                  (0.247, 0.243, 0.261))])
    train = torchvision.datasets.CIFAR10(root=save_path, train=True, transform=transforms, download=True)
    test = torchvision.datasets.CIFAR10(root=save_path, train=False, transform=transforms, download=True)

    torch.save(train, 'data/processed/train.pt')
    torch.save(test, 'data/processed/test.pt')



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

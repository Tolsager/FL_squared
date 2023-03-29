# import os
import click
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import utils
from src.data import process_data
from src.models import model, resnet, simsiam

load_dotenv(find_dotenv())

GPU = torch.cuda.is_available()


@click.group()
def cli():
    pass


import torch
import torchvision

from src import data, engine, model


def train():
    # get datasets
    # train_ds = data.get_Vcifar10_dataset(train=True)
    val_ds = data.get_cifar10_dataset(train=False)

    # cifar10_train_transforms = torchvision.transforms.Compose(
    #     data.cifar10_augmentations
    # )
    # cifar10_val_transforms = torchvision.transforms.Compose(
    #     data.cifar10_standard_transforms
    # )
    # train_ds = data.SimSiamDataset(train_ds, cifar10_train_transforms)
    # train_ds = data.SimSiamDataset(train_ds, data.train_transforms)
    val_ds = data.AugmentedDataset(val_ds, data.base_transforms)

    # create dataloaders
    train_ds = torchvision.datasets.CIFAR10(
        "data/raw",
        train=True,
        download=True,
        transform=data.TwoCropsTransform(data.train_transforms),
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=512, num_workers=8, pin_memory=True, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=512, num_workers=8, pin_memory=True
    )

    simsiam = model.SimSiam()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    simsiam.to(device)
    # trainer = engine.Trainer(
    #     train_dl, val_dl, simsiam, epochs=100, learning_rate=0.06, device=device
    # )
    trainer = engine.Trainer(
        train_dl, val_dl, simsiam, epochs=300, learning_rate=0.06, device=device
    )
    trainer.train()


if __name__ == "__main__":
    cli()

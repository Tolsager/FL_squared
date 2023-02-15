# import os

import click
import torch
from dotenv import find_dotenv, load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import utils
from src.data import process_data
from src.models import model

load_dotenv(find_dotenv())


@click.group()
def cli():
    pass


@click.command()
@click.option("--config-file", type=str, default="debug_train")
def train_fl(config_file: str):
    config = utils.load_config(config_file)
    tags = ["debug"]
    wandb.init(config=config, project="rep-in-fed", entity="pydqn", tags=tags)
    config = wandb.config
    config.server.n_clients
    server = model.Server(**config["server"])
    server.train()


@click.command()
@click.option("--learning_rate", type=float, default=0.001)
@click.option("--batch_size", type=int, default=16)
@click.option("--seed", type=int, default=0)
@click.option("--epochs", type=int, default=5)
def trainbl(learning_rate: float, batch_size: int, seed: int, epochs: int):
    utils.seed_everything(seed)

    logger = WandbLogger(project="rep-in-fed", entity="pydqn",
                         notes="simpnet baseline with augmentations, reduced dropout probabilities, groupnorm after all layers")
    transforms = process_data.get_cifar10_transforms()

    baseline = model.ClientCNN(learning_rate=learning_rate)
    train, test = model.make_dataset.load_dataset()
    train = process_data.AugmentedDataset(train, transforms)
    trainloader_bl = torch.utils.data.DataLoader(train, batch_size=batch_size)
    train, val = process_data.train_val_split(test, 0.2)
    valloader_bl = torch.utils.data.DataLoader(val, batch_size=batch_size)
    trainer = Trainer(accelerator="gpu", gpus=1, max_epochs=epochs, logger=logger)
    trainer.fit(baseline, trainloader_bl, valloader_bl)


cli.add_command(train_fl)
cli.add_command(trainbl)

if __name__ == "__main__":
    cli()

# import os

import click
from dotenv import find_dotenv, load_dotenv

import torch
import wandb
from pytorch_lightning import Trainer
from src import utils
from src.models import model
from src.data import process_data

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
@click.option("--batch_size", type=int, default=10)
@click.option("--seed", type=int, default=0)
@click.option("--epochs", type=int, default=1)
def trainbl(learning_rate: float, batch_size: int, seed: int, epochs: int):
    utils.seed_everything(seed)

    wandb.init(project="rep-in-fed", entity="pydqn")
    wandb_config = wandb.config

    baseline = model.ClientCNN(learning_rate=learning_rate)
    train, test = model.make_dataset.load_dataset()
    trainloader_bl = torch.utils.data.DataLoader(train, batch_size=batch_size)
    val, _ = process_data.val_test_split(test, 0.2)
    valloader_bl = torch.utils.data.DataLoader(val, batch_size=batch_size)
    trainer = Trainer(accelerator="gpu", gpus=1, max_epochs=epochs)
    trainer.fit(baseline, trainloader_bl, valloader_bl)


cli.add_command(train_fl)
cli.add_command(trainbl)

if __name__ == "__main__":
    cli()

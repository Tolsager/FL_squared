# import os

import click
from dotenv import find_dotenv, load_dotenv

import wandb
from src import utils
from src.models import model


@click.command()
@click.option("--config-file", type=str, default="debug_train")
def train_fl(config_file: str):
    config = utils.load_config(config_file)
    tags = ["debug"]
    wandb.init(config=config, project="rep-in-fed", entity="pydqn", tags=tags)
    server = model.Server(**config["server"])
    server.train()


def train_bl():
    pass


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    train_fl()

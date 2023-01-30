# import os

import click

# import wandb
from dotenv import find_dotenv, load_dotenv

from src import utils
from src.models import model


@click.command()
@click.option("--config-file", type=str, default="debug_train")
def train_fl(config_file: str):
    config = utils.load_config(config_file)
    server = model.Server(**config["server"])
    server.train_clients()


def train_bl():
    pass


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    train_fl()

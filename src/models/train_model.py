# import os

import click

# import wandb
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.option("--config_file", type=str, default="train_cpu.yaml")
def train_fl():
    pass


def train_bl():
    pass


if __name__ == "__main__":
    load_dotenv(find_dotenv())

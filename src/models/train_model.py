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
from src.models import model, simsiam

load_dotenv(find_dotenv())

GPU = torch.cuda.is_available()


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


@click.command(name="baseline")
@click.option("--learning-rate", type=float, default=0.001)
@click.option("--batch-size", type=int, default=16)
@click.option("--seed", type=int, default=0)
@click.option("--epochs", type=int, default=5)
def train_bl(learning_rate: float, batch_size: int, seed: int, epochs: int):
    utils.seed_everything(seed)

    logger = WandbLogger(
        project="rep-in-fed",
        entity="pydqn",
        notes="simpnet baseline with less augmentations,\
        reduced dropout probabilities, groupnorm after all layers",
    )
    transforms = process_data.get_cifar10_transforms()

    # baseline = model.SimpNet(embedding_size=10, learning_rate=learning_rate)
    baseline = model.ClientCNN(learning_rate=learning_rate)
    train, test = model.make_dataset.load_dataset()
    train, val = process_data.train_val_split(test, 0.2)
    train = process_data.AugmentedDataset(train, transforms)
    trainloader_bl = torch.utils.data.DataLoader(train, batch_size=batch_size)
    valloader_bl = torch.utils.data.DataLoader(val, batch_size=batch_size)
    trainer = Trainer(accelerator="gpu", gpus=1, max_epochs=epochs, logger=logger)
    trainer.fit(baseline, trainloader_bl, valloader_bl)


@click.command(name="simsiam")
@click.option("--learning-rate", type=float, default=0.001)
@click.option("--batch-size", type=int, default=32)
@click.option("--seed", type=int, default=0)
@click.option("--epochs", type=int, default=2)
@click.option("--embedding-size", type=int, default=2048)
@click.option("--debug", is_flag=True)
@click.option("--backbone", default="resnet")
@click.option("--linear-lr", is_flag=True)
@click.option("--trials", type=int, default=1)
def train_simsiam(
    learning_rate: float,
    batch_size: int,
    seed: int,
    epochs: int,
    embedding_size: int,
    debug: bool,
    backbone: str,
    linear_lr: bool,
    trials: int,
):
    backbones = {"resnet", "simpnet"}
    if backbone not in backbones:
        raise ValueError(
            f"Backbone {backbone} is not supported\n available \
                backbones are: f{backbones}"
        )
    utils.seed_everything(seed)

    if linear_lr:
        learning_rate = 0.03 * batch_size / 256

    train, test = model.make_dataset.load_dataset()

    transforms = process_data.get_simsiam_transforms(img_size=32)
    train, val = process_data.train_val_split(train, 0.2)
    val = process_data.AugmentedDataset(
        val,
        torchvision.transforms.transforms.Compose(
            process_data.cifar10_standard_transforms
        ),
    )
    train = process_data.SimSiamDataset(train, transforms)
    trainloader_bl = torch.utils.data.DataLoader(train, batch_size=batch_size)
    valloader_bl = torch.utils.data.DataLoader(val, batch_size=batch_size)

    for trial in range(trials):
        seed = seed + trial
        utils.seed_everything(seed)

        if not debug:
            tags = ["simsiam", "high learning rate"]
            notes = "memory test"
            # notes = input("Please provide a description of the experiment:\n")
            logger = WandbLogger(
                project="rep-in-fed", entity="pydqn", tags=tags, notes=notes
            )

        if backbone == "resnet":
            backbone_model = torchvision.models.resnet18()
            backbone_model.conv1 = torch.nn.Conv2d(
                kernel_size=3, padding=1, stride=1, in_channels=3, out_channels=64, bias=False
            )
            backbone_model.maxpool = torch.nn.Identity()
            projector = torch.nn.Sequential(
                torch.nn.Linear(
                    backbone_model.fc.in_features, embedding_size, bias=False
                ),
                torch.nn.BatchNorm1d(embedding_size),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_size, embedding_size, bias=False),
                torch.nn.BatchNorm1d(embedding_size, affine=False),
            )
            # backbone_model.fc = torch.nn.Linear(
            #     in_features=backbone_model.fc.in_features, out_features=embedding_size
            # )
            backbone_model.fc = torch.nn.Identity()
        elif backbone == "simpnet":
            backbone_model = model.SimpNet(
                embedding_size=embedding_size, learning_rate=learning_rate
            )

        predictor = simsiam.get_simsiam_predictor(
            embedding_dim=embedding_size, hidden_dim=512
        )
        simsiam_model = simsiam.OurSimSiam(
            backbone=backbone_model,
            projector=projector,
            predictor=predictor,
            learning_rate=learning_rate,
            weight_decay=0.0005,
            max_epochs=epochs,
        )
        if debug:
            trainer = Trainer(limit_train_batches=2, limit_val_batches=2)
        else:
            trainer = (
                Trainer(
                    accelerator="gpu",
                    gpus=1,
                    max_epochs=epochs,
                    logger=logger,
                    callbacks=[simsiam.TrainFeatures()]
                )
                if GPU
                else Trainer(max_epochs=epochs, logger=logger)
            )

        trainer.fit(simsiam_model, trainloader_bl, valloader_bl)
        if not debug:
            wandb.finish()


@click.command(name="imagenet")
@click.option("--learning-rate", type=float, default=0.001)
@click.option("--batch-size", type=int, default=32)
@click.option("--seed", type=int, default=0)
@click.option("--epochs", type=int, default=2)
@click.option("--embedding-size", type=int, default=1000)
@click.option("--arch", type=str, default="simpnet")
def train_imagenet(
    learning_rate: float,
    batch_size: int,
    seed: int,
    epochs: int,
    embedding_size: int,
    arch: str,
):
    utils.seed_everything(seed)

    tags = ["supervised_learning", f"{arch}", "imagenet"]

    logger = WandbLogger(project="rep-in-fed", entity="pydqn", tags=tags)
    if arch == "simpnet":
        simpnet_model = model.SimpNet(embedding_size, learning_rate=learning_rate)

    train = torchvision.datasets.ImageFolder(
        "data/raw/imagenet/train",
        transform=torchvision.transforms.Compose(
            process_data.imagenet_standard_transforms
        ),
    )
    train = process_data.shuffle_dataset(train)
    val = torchvision.datasets.ImageFolder(
        "data/raw/imagenet/val",
        transform=torchvision.transforms.Compose(
            process_data.imagenet_standard_transforms
        ),
    )
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    valloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    trainer = Trainer(
        accelerator="gpu", gpus=1, max_epochs=epochs, logger=logger
    )  # Not feasible to use a non GPU machine to train
    trainer.fit(simpnet_model, trainloader, valloader)


cli.add_command(train_fl)
cli.add_command(train_bl)
cli.add_command(train_simsiam)
cli.add_command(train_imagenet)

if __name__ == "__main__":
    cli()

# import os
import click
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import torch
import torchvision

from src.models import model, resnet, simsiam
from src.data import make_dataset, process_data

load_dotenv(find_dotenv())

GPU = torch.cuda.is_available()


@click.command(name="simsiam")
@click.option("--batch_size", default=512, type=int)
@click.option("--epochs", default=800, type=int)
@click.option("--learning_rate", default=0.06, type=float)
@click.option("--embedding-size", default=2048, type=int)
@click.option("--backbone", default="resnet18", type=str)
@click.option("--num_workers", default=8, type=int)
def train(
        batch_size: int,
        epochs: int,
        learning_rate: float,
        embedding_size: int,
        backbone: str,
        num_workers: int,
):
    architectures = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
    if not (backbone in architectures):
        raise ValueError(f"Architecture {backbone} is not supported must be in {architectures}")

    train_transforms = process_data.get_simsiam_transforms(img_size=32)
    val_transforms = torchvision.transforms.Compose(process_data.cifar10_standard_transforms)

    train_ds = make_dataset.get_cifar10_dataset(root=Path("data/raw"), train=True,
                                                transforms=process_data.TwoCropsTransform(val_transforms, train_transforms))

    val_ds = make_dataset.get_cifar10_dataset(root=Path("data/raw"), train=False, transforms=val_transforms)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    simsiam_model = simsiam.SimSiam(embedding_size=embedding_size)

    device = "cuda" if GPU else "cpu"
    print(f"Training on: {device}")

    simsiam_model.to(device)

    trainer = simsiam.Trainer(
        train_dl, val_dl, simsiam_model, epochs=epochs, learning_rate=learning_rate, device=device
    )
    trainer.train()


@click.group()
def cli():
    pass


cli.add_command(train)

if __name__ == "__main__":
    cli()

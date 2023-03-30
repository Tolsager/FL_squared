# import os
import click
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv

# import wandb
from src.data import make_dataset, process_data
from src.models import simsiam

load_dotenv(find_dotenv())

GPU = torch.cuda.is_available()


@click.group()
def cli():
    pass


@click.command("simsiam")
def train():
    # get datasets
    train_ds, val_ds = make_dataset.load_dataset(dataset="cifar10")

    train_ds = process_data.SimSiamDataset(
        train_ds, process_data.get_simsiam_transforms()
    )
    val_ds = process_data.AugmentedDataset(
        val_ds, torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS)
    )

    # create dataloaders
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=512, num_workers=8, pin_memory=True, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=512, num_workers=8, pin_memory=True
    )

    simsiam_model = simsiam.SimSiam()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    simsiam_model.to(device)
    trainer = simsiam.Trainer(
        train_dl,
        val_dl,
        simsiam_model,
        epochs=300,
        learning_rate=0.06,
        device=device,
        validation_interval=1,
    )
    trainer.train()


cli.add_command(train)

if __name__ == "__main__":
    cli()

import copy

import click
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv

import wandb
from src import utils
from src.data import make_dataset, process_data
from src.models import federated_learning as fl
from src.models import resnet, simsiam

load_dotenv(find_dotenv())

GPU = torch.cuda.is_available()


@click.command(name="federated")
@click.option("--batch-size", default=512, type=int)
@click.option("--epochs", default=4, type=int)
@click.option("--learning-rate", default=0.005, type=float)
@click.option("--backbone", default="resnet18", type=str)
@click.option("--num-workers", default=8, type=int)
@click.option("--log", is_flag=True, default=False)
@click.option(
    "--iid", is_flag=True, default=False, help="if the data is iid or non-iid"
)
@click.option(
    "--val-frac", default=0.1, type=float, help="fraction of data used for validation"
)
@click.option("--seed", default=0, type=int)
@click.option("--n-clients", default=10, type=int)
@click.option("--n-rounds", default=10, type=int)
def train_federated(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    backbone: str,
    num_workers: int,
    log: bool,
    iid: bool,
    val_frac: float,
    seed: int,
    n_clients: int,
    n_rounds: int,
):
    tags = ["debug"]
    utils.seed_everything(seed)
    train_ds, test_ds = make_dataset.load_dataset(dataset="cifar10")

    val_dl = None
    if val_frac > 0:
        # stratified validation split
        train_ds, val_ds = process_data.stratified_train_val_split(
            train_ds, label_fn=process_data.cifar10_sort_fn, val_size=val_frac
        )
        val_ds = process_data.AugmentedDataset(
            val_ds,
            torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS),
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )

    train_transforms = torchvision.transforms.Compose(
        process_data.CIFAR10_SUPERVISED_TRANSFORMS
    )
    train_ds = process_data.AugmentedDataset(train_ds, train_transforms)

    # sort train_ds
    train_ds = process_data.sort_dataset(train_ds, process_data.cifar10_sort_fn)

    # split the data to the clients
    train_datasets = process_data.simple_datasplit(train_ds, n_clients)
    client_dataloaders = [
        torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        for ds in train_datasets
    ]

    fl_model = resnet.ResNet18Classifier(n_classes=10)

    # instantiate the client models
    client_models = [copy.deepcopy(fl_model) for _ in range(n_clients)]

    optimizer = torch.optim.SGD
    criterion = torch.nn.CrossEntropyLoss()

    device = "cuda" if GPU else "cpu"

    print(f"Training on: {device}")

    wandb.init(
        project="rep-in-fed",
        entity="pydqn",
        mode="online" if log else "disabled",
        tags=tags,
    )
    trainer = fl.SupervisedTrainer(
        client_dataloaders,
        val_dl,
        client_models,
        epochs=epochs,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        rounds=n_rounds,
        learning_rate=learning_rate,
    )
    trainer.train()


@click.command(name="simsiam")
@click.option("--batch-size", default=512, type=int)
@click.option("--epochs", default=5, type=int)
@click.option("--learning-rate", default=0.06, type=float)
@click.option(
    "--val_frac", default=0.2, type=float, help="fraction of data used for validation"
)
@click.option("--embedding-size", default=2048, type=int)
@click.option("--backbone", default="resnet18", type=str)
@click.option("--num-workers", default=8, type=int)
@click.option("--min-scale", default=0.2, type=float)
@click.option("--log", is_flag=True, default=False)
def train_simsiam(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    val_frac: float,
    embedding_size: int,
    backbone: str,
    num_workers: int,
    min_scale: float,
    log: bool,
):
    architectures = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
    if not (backbone in architectures):
        raise ValueError(
            f"Architecture {backbone} is not supported must be in {architectures}"
        )

    train_ds, test_ds = make_dataset.load_dataset(dataset="cifar10")

    val_dl = None
    if val_frac > 0:
        train_ds, val_ds = process_data.stratified_train_val_split(
            train_ds, label_fn=process_data.cifar10_sort_fn, val_size=val_frac
        )
        val_ds = process_data.AugmentedDataset(
            val_ds,
            torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS),
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )

    train_ds = process_data.SimSiamDataset(
        train_ds, process_data.get_cifar10_transforms(min_scale=min_scale)
    )

    # create dataloaders
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    simsiam_model = simsiam.SimSiam(embedding_size=embedding_size)

    device = "cuda" if GPU else "cpu"
    print(f"Training on: {device}")

    simsiam_model.to(device)

    wandb.init(
        project="rep-in-fed", entity="pydqn", mode="online" if log else "disabled"
    )
    trainer = simsiam.Trainer(
        train_dl,
        val_dl,
        simsiam_model,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        validation_interval=1,
    )
    trainer.train()


@click.command(name="federated_simsiam")
@click.option("--batch_size", default=512, type=int)
@click.option("--epochs", default=800, type=int)
@click.option("--learning_rate", default=0.06, type=float)
@click.option("--embedding-size", default=2048, type=int)
@click.option("--backbone", default="resnet18", type=str)
@click.option("--num_workers", default=8, type=int)
@click.option(
    "--rounds", default=5, type=int, help="Number of training rounds clients to perform"
)
@click.option("--log", is_flag=True, default=False)
def train_federated_simsiam(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    embedding_size: int,
    backbone: str,
    num_workers: int,
    rounds: int,
    log: bool,
):
    pass


@click.group()
def cli():
    pass


cli.add_command(train_simsiam)
cli.add_command(train_federated_simsiam)
cli.add_command(train_federated)

if __name__ == "__main__":
    cli()

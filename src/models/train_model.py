import click
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv

import wandb
from src import utils
from src.data import make_dataset, process_data
from src.models import federated_learning as fl
from src.models import federated_simsiam as fss
from src.models import model, resnet, simsiam

load_dotenv(find_dotenv())

GPU = torch.cuda.is_available()
DEVICE = "cuda" if GPU else "cpu"


@click.command(name="supervised")
@click.option("--batch-size", default=512, type=int)
@click.option("--epochs", default=100, type=int)
@click.option("--learning-rate", default=0.001, type=float)
@click.option("--backbone", default="resnet18", type=str)
@click.option("--num-workers", default=8, type=int)
@click.option("--log", is_flag=True, default=False)
@click.option("--val-frac", default=0.1, type=float)
@click.option("--seed", default=0, type=int)
def train_supervised(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    backbone: str,
    num_workers: int,
    log: bool,
    val_frac: float,
    seed: int,
):
    tags = ["debug"]
    utils.seed_everything(seed)
    train_ds, test_ds = make_dataset.load_dataset(dataset="cifar10")

    val_dl = None
    if val_frac > 0:
        train_ds = process_data.sort_dataset(train_ds, process_data.cifar10_sort_fn)
        train_ds, val_ds = process_data.stratified_train_val_split(
            train_ds, process_data.cifar10_sort_fn, val_frac
        )

        val_transforms = torchvision.transforms.Compose(
            process_data.CIFAR10_STANDARD_TRANSFORMS
        )

        val_ds = process_data.AugmentedDataset(val_ds, val_transforms)

        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )

    train_transforms = torchvision.transforms.Compose(
        process_data.CIFAR10_SUPERVISED_TRANSFORMS
    )

    train_ds = process_data.AugmentedDataset(train_ds, train_transforms)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    supervised_model = resnet.ResNet18Classifier(n_classes=10)

    print(f"Training on {DEVICE}")

    supervised_model = supervised_model.to(DEVICE)

    wandb.init(
        project="rep-in-fed",
        entity="pydqn",
        mode="online" if log else "disabled",
        tags=tags,
    )

    trainer = model.SupervisedTrainer(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        model=supervised_model,
        epochs=epochs,
        learning_rate=learning_rate,
        device=DEVICE,
    )

    trainer.train()


@click.command(name="federated")
@click.option("--batch-size", default=512, type=int)
@click.option("--epochs", default=4, type=int)
@click.option("--learning-rate", default=0.005, type=float)
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
@click.option("--sweep", is_flag=True, default=False)
@click.option("--total-epochs", type=int, default=40)
def train_federated(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_workers: int,
    log: bool,
    iid: bool,
    val_frac: float,
    seed: int,
    n_clients: int,
    n_rounds: int,
    sweep: bool,
    total_epochs: int,
):
    if sweep:
        epoch_map = {1: 4, 2: 5, 3: 8, 4: 10, 5: 20}
        epochs = epoch_map[epochs]
        n_rounds = total_epochs // epochs

    config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_clients": n_clients,
        "n_rounds": n_rounds,
    }
    tags = ["fedavg"]
    notes = "Finding maximum learning rate before divergence"
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
    if iid:
        train_datasets = process_data.simple_datasplit(train_ds, n_clients)
    else:
        datasplitter = process_data.DataSplitter(
            train_ds, n_clients, shards_per_client=2
        )
        train_datasets = datasplitter.split_data()

    client_dataloaders = [
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
        for ds in train_datasets
    ]

    fl_model = resnet.ResNet18Classifier(n_classes=10)

    optimizer = torch.optim.SGD
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Training on: {DEVICE}")

    wandb.init(
        project="rep-in-fed",
        entity="pydqn",
        mode="online" if log else "disabled",
        tags=tags,
        notes=notes,
        config=config,
    )
    trainer = fl.SupervisedTrainer(
        client_dataloaders,
        val_dl,
        fl_model,
        epochs=epochs,
        device=DEVICE,
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
    "--val-frac", default=0.1, type=float, help="fraction of data used for validation"
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
    utils.seed_everything(0)
    config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    tags = ["simsiam"]
    notes = "find optimal learning rate"
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

    print(f"Training on: {DEVICE}")

    simsiam_model.to(DEVICE)

    wandb.init(
        project="rep-in-fed",
        entity="pydqn",
        mode="online" if log else "disabled",
        config=config,
        notes=notes,
        tags=tags,
    )
    trainer = simsiam.Trainer(
        train_dl,
        val_dl,
        simsiam_model,
        epochs=epochs,
        learning_rate=learning_rate,
        device=DEVICE,
        validation_interval=1,
    )
    trainer.train()


@click.command(name="FL2")
@click.option("--batch-size", default=512, type=int)
@click.option("--epochs", default=5, type=int)
@click.option("--learning-rate", default=0.06, type=float)
@click.option("--embedding-size", default=2048, type=int)
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
@click.option(
    "--rounds", default=5, type=int, help="Number of training rounds clients to perform"
)
@click.option("--validation-interval", default=1, type=int)
@click.option("--sweep", is_flag=True, default=False)
@click.option("--total-epochs", type=int, default=40)
def train_federated_simsiam(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    embedding_size: int,
    backbone: str,
    num_workers: int,
    log: bool,
    iid: bool,
    val_frac: float,
    seed: int,
    n_clients: int,
    rounds: int,
    validation_interval: int,
    sweep: bool,
    total_epochs: int,
):
    if sweep:
        epoch_map = {1: 4, 2: 5, 3: 8, 4: 10, 5: 20}
        epochs = epoch_map[epochs]
        rounds = total_epochs // epochs
    utils.seed_everything(seed)
    config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    tags = ["FL2"]
    notes = "train with 80% of the data"
    architectures = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
    if not (backbone in architectures):
        raise ValueError(
            f"Architecture {backbone} is not supported must be in {architectures}"
        )

    train_ds, test_ds = make_dataset.load_dataset(dataset="cifar10")

    val_dl = None
    if val_frac > 0:
        # supervised frac and val frac are assumed to be the same
        train_ds, val_and_supervised_ds = process_data.stratified_train_val_split(
            train_ds, label_fn=process_data.cifar10_sort_fn, val_size=val_frac * 2
        )

        val_and_supervised_ds = process_data.AugmentedDataset(
            val_and_supervised_ds,
            torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS),
        )

        val_ds, supervised_ds = process_data.stratified_train_val_split(
            val_and_supervised_ds, process_data.cifar10_sort_fn, 0.5
        )
        val_dl = torch.utils.data.DataLoader(val_ds, pin_memory=True)

    # split the data to the clients
    if iid:
        train_datasets = process_data.simple_datasplit(train_ds, n_clients)
    else:
        datasplitter = process_data.DataSplitter(
            train_ds, n_clients, shards_per_client=2
        )
        train_datasets = datasplitter.split_data()

    train_ds = process_data.SimSiamDataset(
        train_ds, process_data.get_cifar10_transforms()
    )

    # sort train_ds
    train_ds = process_data.sort_dataset(train_ds, process_data.simsiam_sort_fn)

    # split the data to the clients
    if iid:
        train_datasets = process_data.simple_datasplit(train_ds, n_clients)
    else:
        datasplitter = process_data.DataSplitter(
            train_ds, n_clients, shards_per_client=2
        )
        train_datasets = datasplitter.split_data()

    client_dataloaders = [
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
        for ds in train_datasets
    ]

    model = simsiam.SimSiam(embedding_size=embedding_size)
    optimizer = torch.optim.SGD

    trainer = fss.FedAvgSimSiamTrainer(
        client_dataloaders,
        val_dl,
        model,
        optimizer,
        rounds,
        epochs,
        device=DEVICE,
        learning_rate=learning_rate,
        validation_interval=validation_interval,
    )
    wandb.init(
        project="rep-in-fed",
        entity="pydqn",
        mode="online" if log else "disabled",
        config=config,
        notes=notes,
        tags=tags,
    )
    trainer.train()


@click.command(name="FLS")
@click.option("--batch-size", default=512, type=int)
@click.option("--local-epochs", default=4, type=int)
@click.option("--supervised-epochs", default=5, type=int)
@click.option("--learning-rate", default=0.06, type=float)
@click.option("--supervised-learning-rate", default=0.06, type=float)
@click.option("--embedding-size", default=2048, type=int)
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
@click.option(
    "--rounds", default=5, type=int, help="Number of training rounds clients to perform"
)
def train_federated_supervised_simsiam(
    batch_size: int,
    local_epochs: int,
    supervised_epochs: int,
    learning_rate: float,
    supervised_learning_rate: float,
    embedding_size: int,
    backbone: str,
    num_workers: int,
    log: bool,
    iid: bool,
    val_frac: float,
    seed: int,
    n_clients: int,
    rounds: int,
):
    utils.seed_everything(seed)
    config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    tags = ["debug"]
    notes = "find optimal learning rate"
    architectures = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
    if not (backbone in architectures):
        raise ValueError(
            f"Architecture {backbone} is not supported must be in {architectures}"
        )

    train_ds, test_ds = make_dataset.load_dataset(dataset="cifar10")

    val_dl = None
    if val_frac > 0:
        # supervised frac and val frac are assumed to be the same
        train_ds, val_and_supervised_ds = process_data.stratified_train_val_split(
            train_ds, label_fn=process_data.cifar10_sort_fn, val_size=val_frac * 2
        )

        val_and_supervised_ds = process_data.AugmentedDataset(
            val_and_supervised_ds,
            torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS),
        )

        # sort the dataset for validation and supervised training and split it in 2
        val_and_supervised_ds = process_data.sort_dataset(
            val_and_supervised_ds, process_data.cifar10_sort_fn
        )
        val_ds, supervised_ds = process_data.stratified_train_val_split(
            val_and_supervised_ds, process_data.cifar10_sort_fn, 0.5
        )
        val_dl = torch.utils.data.DataLoader(val_ds, pin_memory=True)

    # split the data to the clients
    if iid:
        train_datasets = process_data.simple_datasplit(train_ds, n_clients)
    else:
        datasplitter = process_data.DataSplitter(
            train_ds, n_clients, shards_per_client=2
        )
        train_datasets = datasplitter.split_data()

    supervised_dl = torch.utils.data.DataLoader(
        supervised_ds, shuffle=True, pin_memory=True
    )

    train_datasets = [
        process_data.SimSiamDataset(ds, process_data.get_cifar10_transforms())
        for ds in train_datasets
    ]

    client_dataloaders = [
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
        for ds in train_datasets
    ]

    model = simsiam.SimSiam(embedding_size=embedding_size)
    optimizer = torch.optim.SGD

    trainer = fss.FedAvgSimSiamFinetuningTrainer(
        client_dataloaders,
        supervised_dl,
        val_dl,
        model,
        optimizer,
        rounds,
        local_epochs,
        supervised_epochs,
        device=DEVICE,
        learning_rate=learning_rate,
        supervised_learning_rate=supervised_learning_rate,
    )
    wandb.init(
        project="rep-in-fed",
        entity="pydqn",
        mode="online" if log else "disabled",
        config=config,
        notes=notes,
        tags=tags,
    )
    trainer.train()


@click.group()
def cli():
    pass


cli.add_command(train_supervised)
cli.add_command(train_simsiam)
cli.add_command(train_federated_simsiam)
cli.add_command(train_federated)
cli.add_command(train_federated_supervised_simsiam)

if __name__ == "__main__":
    cli()

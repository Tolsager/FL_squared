import os
from datetime import datetime
from typing import *

import click
import flwr as fl
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv

import wandb
from src import utils
from src.data import make_dataset, process_data
from src.models import fed_flwr
from src.models import federated_simsiam as fss
from src.models import model, resnet, simsiam

# os.environ["RAY_memory_monitor_refresh_ms"] = "0"

load_dotenv(find_dotenv())

GPU = torch.cuda.is_available()
DEVICE = "cuda" if GPU else "cpu"


@click.command(name="supervised")
@click.option("--batch-size", default=512, type=int)
@click.option("--local-epochs", default=4, type=int)
@click.option("--lr", default=0.005, type=float)
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
def supervised(
    batch_size,
    local_epochs,
    lr,
    num_workers,
    log,
    iid,
    val_frac,
    seed,
    n_clients,
    n_rounds,
):
    config = {
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "lr": lr,
        "n_clients": n_clients,
        "n_rounds": n_rounds,
    }
    wandb.init(
        project="rep-in-fed",
        entity="pydqn",
        mode="online" if log else "disabled",
        config=config,
    )
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
            pin_memory=False,
            shuffle=True,
        )
        for ds in train_datasets
    ]

    def client_fn(cid: str):
        net = resnet.ResNet18Classifier(n_classes=10)
        net.to(DEVICE)
        train_dl = client_dataloaders[int(cid)]
        return fed_flwr.CifarClient(net, train_dl, local_epochs, lr, DEVICE)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    best_val_acc = 0

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        global best_val_acc
        net = resnet.ResNet18Classifier(10)
        net.to(DEVICE)
        fed_flwr.set_parameters(
            net, parameters
        )  # Update model with the latest parameters
        loss, metrics = fed_flwr.validate(net, val_dl, DEVICE)
        wandb.log({"val_loss": loss, "val_acc": metrics["accuracy"]})

        print(f"Server loss: {loss}, server accuracy: {metrics['accuracy']}")

        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            torch.save(
                model.state_dict(),
                f"models/fedavg_{'iid_' if iid else 'non_iid_'}{timestamp}.pth",
            )

        return loss, metrics

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1, evaluate_fn=evaluate, fraction_evaluate=0
    )
    client_resources = {"num_gpus": 1}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )


@click.group()
def cli():
    pass


cli.add_command(supervised)

if __name__ == "__main__":
    cli()

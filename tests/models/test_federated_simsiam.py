import copy

import torch
import torchvision

import wandb
from src.data import make_dataset, process_data
from src.debug_utils import are_models_identical
from src.models import federated_simsiam as fls
from src.models import simsiam

train_ds, test_ds = make_dataset.load_dataset()


def test_trainer():
    n_clients = 2
    batch_size = 50
    n_batches = 2
    learning_rate = 3e-5
    epochs = 2
    n_rounds = 2
    device = "cuda"

    global train_ds

    val_subset = torch.utils.data.Subset(train_ds, [*range(n_batches * batch_size)])
    val_subset = process_data.AugmentedDataset(
        val_subset,
        torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS),
    )
    val_dl = torch.utils.data.DataLoader(val_subset, batch_size=2)
    train_ds = process_data.SimSiamDataset(
        train_ds, process_data.get_cifar10_transforms()
    )
    train_subset = torch.utils.data.Subset(
        train_ds, [*range(n_clients * n_batches * batch_size)]
    )
    train_datasets = process_data.simple_datasplit(train_subset, n_clients)
    client_dataloaders = [
        torch.utils.data.DataLoader(ds, batch_size=batch_size) for ds in train_datasets
    ]

    model = simsiam.SimSiam(embedding_size=2048)

    optimizer = torch.optim.SGD

    wandb.init(project="rep-in-fed", entity="pydqn", mode="disabled")
    trainer = fls.FedAvgSimSiamTrainer(
        client_dataloaders,
        val_dl,
        model,
        epochs=epochs,
        device=device,
        optimizer=optimizer,
        rounds=n_rounds,
        learning_rate=learning_rate,
        validation_interval=1,
    )
    new_fl_model = trainer.train()

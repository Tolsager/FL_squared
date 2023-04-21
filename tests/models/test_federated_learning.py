import copy

import torch
import torchvision

import wandb
from src.data import make_dataset, process_data
from src.models import federated_learning as fl
from src.models import resnet

train_ds, test_ds = make_dataset.load_dataset()
train_ds = process_data.AugmentedDataset(
    train_ds, torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS)
)


def test_supervised_trainer():
    n_clients = 2
    batch_size = 2
    n_batches = 2
    learning_rate = 3e-5
    epochs = 2
    rounds = 2
    device = "cuda"

    train_subset = torch.utils.data.Subset(
        train_ds, [*range(n_clients * n_batches * batch_size)]
    )
    train_datasets = process_data.simple_datasplit(train_subset, n_clients)
    client_dataloaders = [
        torch.utils.data.DataLoader(ds, batch_size=batch_size) for ds in train_datasets
    ]
    val_subset = torch.utils.data.Subset(train_ds, [*range(n_batches * batch_size)])
    val_dl = torch.utils.data.DataLoader(val_subset, batch_size=2)

    fl_model = resnet.ResNet18Classifier(n_classes=10)

    # instantiate the client models
    client_models = [copy.deepcopy(fl_model) for _ in range(n_clients)]
    for m in client_models:
        m.to(device)

    client_optimizers = [
        torch.optim.AdamW(m.parameters(), lr=learning_rate) for m in client_models
    ]

    criterion = torch.nn.CrossEntropyLoss()

    wandb.init(project="rep-in-fed", entity="pydqn", mode="disabled")
    trainer = fl.SupervisedTrainer(
        client_dataloaders,
        val_dl,
        client_models,
        epochs=epochs,
        device=device,
        optimizer=client_optimizers,
        criterion=criterion,
        rounds=rounds,
    )
    trainer.train()

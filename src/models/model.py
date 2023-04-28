import copy
import random
from typing import List, Tuple, Optional

import resnet

import torch
import torch.nn as nn
import torchmetrics
import tqdm

import wandb
from src.data import make_dataset, process_data


class Server:
    def __init__(
        self,
        n_clients: int,
        shards_per_client: int = 2,
        rounds: int = 2,
        C: float = 0.1,
        batch_size: int = 32,
        n_samples: int = None,
        E: int = 2,
    ):
        self.n_clients = n_clients
        self.rounds = rounds
        self.E = E

        self.server_side_model = ClientCNN()
        self.client_list = None
        self.training_set, self.test_set = make_dataset.load_dataset(
            n_samples=n_samples
        )
        self.data_splitter = process_data.DataSplitter(
            self.training_set, self.n_clients, shards_per_client
        )
        self.client_datasets = self.data_splitter.split_data()
        self.C = C
        self.m = max(int(self.n_clients * self.C), 1)
        self.batch_size = batch_size

        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=batch_size
        )
        self.test_trainer = (
            Trainer(accelerator="gpu", devices=1, enable_model_summary=False)
            if torch.cuda.is_available()
            else Trainer(enable_model_summary=False)
        )
        self.client_trainers = []

    def generate_clients(self):
        return [copy.deepcopy(self.server_side_model) for _ in range(self.n_clients)]

    def do_round(self):
        # TODO: can be optimized to only update the models that are going to be trained
        self.client_list = self.generate_clients()

        self.client_trainers = []

        clients_to_train = random.sample(range(self.n_clients), self.m)

        # Amount of samples used to train the clients in the current round
        data_size = sum([len(self.client_datasets[i]) for i in clients_to_train])

        # Train each client sequentially, TODO: parallelization
        for client_idx in clients_to_train:
            client_trainer = (
                Trainer(
                    max_epochs=self.E,
                    accelerator="gpu",
                    devices=1,
                    enable_model_summary=False,
                )
                if torch.cuda.is_available()
                else Trainer(max_epochs=self.E, enable_model_summary=False)
            )
            self.client_trainers.append(client_trainer)
            client = self.client_list[client_idx]
            client_dataset = self.client_datasets[client_idx]
            client_loader = torch.utils.data.DataLoader(
                client_dataset, batch_size=self.batch_size
            )
            client_trainer.fit(client, client_loader)

        # Average the weights of the clients
        avg_state_dict = self.get_neutral_state_dict()
        for client_idx in clients_to_train:
            for key in avg_state_dict.keys():
                weighted_params = self.client_list[client_idx].state_dict()[key] * len(
                    self.client_datasets[client_idx]
                )
                avg_state_dict[key] = avg_state_dict[key] + weighted_params

        for key in avg_state_dict.keys():
            avg_state_dict[key] /= data_size

        return avg_state_dict

    def train(self):
        for _ in range(self.rounds):
            client_state_dict = self.do_round()
            self.server_side_model.load_state_dict(client_state_dict)
            test_metrics = self.test_trainer.test(
                self.server_side_model, self.test_loader
            )
            wandb.log(test_metrics[0])

    @staticmethod
    def get_neutral_state_dict():
        state_dict = ClientCNN().state_dict()
        for key in state_dict.keys():
            state_dict[key] = torch.zeros_like(state_dict[key])
        return state_dict


class ClientCNN:
    pass


class SupervisedTrainer:
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        model: torch.nn.Module,
        epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-2,
        device: str = "cuda",
        validation_interval: int = 5,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(
            model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay
        )
        self.avg_train_loss = torchmetrics.MeanMetric()
        self.criterion = nn.CrossEntropyLoss()
        self.validation_interval = validation_interval
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)

    def train_epoch(self) -> None:
        self.model.train()
        for img, label in self.train_dataloader:
            img = img.to(self.device)
            label = label.to(self.device)

            output = self.model(img)
            loss = self.criterion(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss = loss.detach().cpu().item()
            self.avg_train_loss.update(loss)

    def train(self) -> None:
        for epoch in tqdm.trange(self.epochs):
            self.avg_train_loss.reset()
            self.train_epoch()
            avg_train_loss = self.avg_train_loss.compute()
            print(f"Epoch: {epoch}")
            print(f"Average train loss: {avg_train_loss}")
            val_acc = self.validation()
            wandb.log({"train_loss": avg_train_loss, "epoch": epoch, "top1_val_acc": val_acc})

        wandb.finish()

    def validation(self) -> float:
        self.model.eval()
        self.val_acc.reset()
        with torch.no_grad():
            for img, label in self.val_dataloader:
                img = img.to(self.device)
                label = label.to(self.device)
                output = self.model(img)
                self.val_acc.update(output.argmax(dim=1), label)

        return self.val_acc.compute().item()


class SupervisedModel(nn.Module):
    def __init__(self, backbone: str = "resnet18", num_classes: int = 10):
        super(SupervisedModel, self).__init__()
        self.backbone = self.get_backbone(backbone)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def get_backbone(backbone_name):
        return {
            "resnet18": resnet.ResNet18(),
            "resnet34": resnet.ResNet34(),
            "resnet50": resnet.ResNet50(),
            "resnet101": resnet.ResNet101(),
            "resnet152": resnet.ResNet152(),
        }[backbone_name]

    def forward(self, img):
        x = self.backbone(img)
        x = self.fc(x)
        return x

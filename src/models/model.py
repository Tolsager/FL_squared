import copy
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy

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

    def generate_clients(self) -> List[LightningModule]:
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


class ClientCNN(LightningModule):
    def __init__(self, learning_rate: float):
        super().__init__()

        self.learning_rate = learning_rate

        self.accuracy = Accuracy(task="multiclass", num_classes=10)

        self.criterion = (
            nn.CrossEntropyLoss()
        )  # TODO: representation learning; contrastive loss
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.block(x)
        return self.out(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(logits, labels)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(logits, labels)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    Server = Server(5, 4)
    Server.train()

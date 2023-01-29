import torch
import random
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np

import copy
from typing import Tuple, List

from src.data import process_data, make_dataset


class Server:
    def __init__(self, num_clients: int, rounds: int = 2):
        self.num_clients = num_clients
        self.rounds = rounds
        self.server_side_model = ClientCNN()
        self.client_list = None
        self.training_set, self.test_set = make_dataset.load_dataset()
        self.data_splitter = process_data.DataSplitter(self.training_set, self.num_clients)
        self.client_datasets = self.data_splitter.split_data()

    def generate_clients(self) -> List[pl.LightningModule]:
        return [copy.deepcopy(self.server_side_model) for _ in range(self.num_clients)]

    def update(self):
        self.client_list = self.generate_clients()
        clients_to_train = random.randint(1, self.num_clients)
        samples = random.sample(range(len(self.client_list)), clients_to_train)

        print(f"Training {clients_to_train} clients.")

        # Amount of samples used to train the clients in the current round
        data_size = np.array([len(self.client_datasets[i]) for i in samples]).sum()

        # Train each client sequentially, TODO: parallelization
        for client_idx in samples:
            client = self.client_list[client_idx]
            client_dataset = self.client_datasets[client_idx]
            client_loader = torch.utils.data.DataLoader(client_dataset, batch_size=32)
            client_trainer = pl.Trainer(max_epochs=self.rounds, gpus=1)
            client_trainer.fit(client, client_loader)

        # Average the weights of the clients
        avg_state_dict = self.get_neutral_state_dict()
        for client_idx in samples:
            for key in avg_state_dict.keys():
                avg_state_dict[key] += self.client_list[client_idx].state_dict()[key] * len(
                    self.client_datasets[client_idx])

        for key in avg_state_dict.keys():
            avg_state_dict[key] /= data_size

        # Update the server-side model
        self.server_side_model.load_state_dict(avg_state_dict)

    def get_neutral_state_dict(self):
        state_dict = copy.deepcopy(self.server_side_model.state_dict())
        for key in state_dict.keys():
            state_dict[key] = torch.zeros_like(state_dict[key])
        return state_dict


class ClientCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()  # TODO: representation learning; contrastive loss
        self.input = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                                   nn.ReLU())
        self.block = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        self.out = nn.Sequential(nn.Flatten(),
                                 nn.Linear(32 * 16 * 16, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.block(x)
        return self.out(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


Server = Server(3)
Server.update()

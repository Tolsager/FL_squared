import torch
import random
import pytorch_lightning as pl
import torch.nn as nn

from src.data import process_data

class Server:
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.datasplitter = process_data.DataSplitter(dataset, num_clients, shards_per_client = 2)
        self.client_list = [ClientCNN(self) for _ in range(num_clients)]

    def update(self):
        clients_to_train = random.randint(1, self.num_clients)
        sample = random.sample(self.client_list, clients_to_train)
        for client in sample:
            pass




class ClientCNN(nn.Module):
    def __init__(self, server: Server):
        super().__init__()
        self.server = server
        self.criterion = nn.CrossEntropyLoss() # TODO: representation learning; contrastive loss
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



Server = Server(3)
print(Server.client_list)
Server.update()

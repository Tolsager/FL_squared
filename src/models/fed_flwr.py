from collections import OrderedDict
from typing import *

import flwr as fl
import numpy as np
import torch
import torchmetrics


def train(
    net: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    device: str,
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SDG(net.parameters(), lr=lr)
    for _ in range(epochs):
        for image, label in train_dl:
            image = image.to(device)

            optimizer.zero_grad()
            logits = net(image)
            label = label.to(device)
            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        net: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        epochs: int,
        lr: float,
        device: str,
    ):
        self.net = net
        self.train_dl = train_dl
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_parameters(self.net)

    def set_parameters(self, parameters):
        set_parameters(self.net, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(config={}), len(self.train_dl), {}

    def train(self):
        # self.net.to(self.device)
        self.net.train()
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.lr, weight_decay=1e-5
        )
        for _ in range(self.epochs):
            for image, label in self.train_dl:
                image = image.to(self.device)

                optimizer.zero_grad()
                logits = self.net(image)
                del image
                label = label.to(self.device)
                loss = self.criterion(logits, label)
                del logits
                del label

                loss.backward()
                optimizer.step()
        # self.net.to("cpu")


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def validate(model: torch.nn.Module, val_dl, device):
    criterion = torch.nn.CrossEntropyLoss()
    avg_loss = torchmetrics.MeanMetric().to(device)
    model.eval()
    # model.to(device)
    val_acc = torchmetrics.Accuracy("multiclass", num_classes=10).to(device)
    with torch.no_grad():
        for image, label in val_dl:
            image = image.to(device)

            logits = model(image)
            del image
            label = label.to(device)
            loss = criterion(logits, label)
            avg_loss(loss)
            out = torch.argmax(logits, dim=1)

            val_acc(out, label)
            del logits
            del out

    val_acc = float(val_acc.compute())
    avg_loss = float(avg_loss.compute())
    # model.to("cpu")
    return avg_loss, {"accuracy": val_acc}

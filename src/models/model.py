import copy
import random
from datetime import datetime
from typing import List, Tuple, Optional

import resnet
import torch
import torch.nn as nn
import torchmetrics
import tqdm

import wandb


class CentralizedTrainer:
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        model: torch.nn.Module,
        epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 0,
        device: str = "cuda",
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(
            model.parameters(), learning_rate, weight_decay=weight_decay
        )
        self.avg_train_loss = torchmetrics.MeanMetric()
        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(
            self.device
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)
        self.best_val_acc = 0.0
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

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
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(), f"Centralized_{self.timestamp}_model.pth"
                )
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
    
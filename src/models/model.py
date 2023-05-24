from datetime import datetime
from typing import Optional

import resnet
import torch
import torch.nn as nn
import torchmetrics
import tqdm

import wandb


class SupervisedTrainer:
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        model: torch.nn.Module,
        epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 0,
        device: str = "cuda",
        unfreeze: bool = False,
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

        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.best_val_acc = 0.0

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
            print(f"Validation accuracy: {val_acc}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(), f"models/Supervised_model_{self.timestamp}.pth")

            wandb.log(
                {"train_loss": avg_train_loss, "epoch": epoch, "val_acc": val_acc}
            )

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


class SupervisedFinetuner(SupervisedTrainer):
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        model: torch.nn.Module,
        epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 0,
        device: str = "cuda",
        unfreeze: bool = False,
        iid: bool = False,
        model_weights: str = None,
    ):
        super().__init__(
            train_dataloader,
            val_dataloader,
            model,
            epochs,
            learning_rate,
            weight_decay,
            device,
        )
        self.unfreeze = unfreeze
        self.model_weights = model_weights
        self.iid = iid

    def train(self) -> None:
        for epoch in tqdm.trange(self.epochs):
            self.avg_train_loss.reset()
            self.train_epoch()
            avg_train_loss = self.avg_train_loss.compute()
            print(f"Epoch: {epoch}")
            print(f"Average train loss: {avg_train_loss}")
            val_acc = self.validation()
            print(f"Validation accuracy: {val_acc}")

            if epoch == 500 and self.unfreeze:
                for param in self.model.parameters():
                    param.requires_grad = True

                self.unfreeze = False

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(), f"models/{'iid_' if self.iid else 'non_iid_'}Finetuned_FLS_{self.timestamp}.pth")

            wandb.log(
                {"train_loss": avg_train_loss, "epoch": epoch, "val_acc": val_acc}
            )

        wandb.finish()




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

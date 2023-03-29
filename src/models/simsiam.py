import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import tqdm
from pytorch_lightning import Callback, LightningModule
from torch import nn

from src.metrics import KNN
from src.models.metrics import KNN


class SimpNetEncoder(nn.Module):
    def __init__(self, embedding_size: int = 432):
        super().__init__()

        self.embedding_size = embedding_size

        self.features = self._make_layers()

        self.fc = nn.Linear(432, embedding_size)

    def _make_layers(self) -> nn.Sequential:
        model = nn.Sequential(
            nn.Conv2d(3, 66, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(11, 66),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.01),
            nn.Conv2d(66, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.02),
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.02),
            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.02),
            nn.Conv2d(128, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(24, 192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            nn.Dropout2d(p=0.04),
            nn.Conv2d(192, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(24, 192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.02),
            nn.Conv2d(192, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(24, 192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.02),
            nn.Conv2d(192, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(24, 192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.025),
            nn.Conv2d(192, 192, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(24, 192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.025),
            nn.Conv2d(192, 288, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(24, 288),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False
            ),
            nn.Dropout2d(p=0.04),
            nn.Conv2d(288, 288, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(24, 288),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.03),
            nn.Conv2d(288, 355, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(71, 355),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.03),
            nn.Conv2d(355, 432, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.GroupNorm(27, 432),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

        return model

    def forward(self, x):
        x = self.features(x)

        # Global Max Pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = F.dropout2d(x, 0.02, training=True)

        x = x.view(x.size(0), -1)
        encoding = self.fc(x)
        return encoding


# def asymmetric_loss(p, z):
#     z = z.detach()  # stop gradient

#     return -torch.nn.functional.cosine_similarity(p, z, dim=-1).mean()


# def criterion(model_outputs: dict):
#     loss1 = asymmetric_loss(model_outputs["p1"], model_outputs["z2"])
#     loss2 = asymmetric_loss(model_outputs["p2"], model_outputs["z1"])
#     return 0.5 * loss1 + 0.5 * loss2
class SimSiamLoss(nn.Module):
    def __init__(self, version="simplified"):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        if self.ver == "original":
            z = z.detach()  # stop gradient

            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)

            return -(p * z).sum(dim=1).mean()

        elif self.ver == "simplified":
            z = z.detach()  # stop gradient
            return -nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):

        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)

        return 0.5 * loss1 + 0.5 * loss2


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args["learning_rate"]
    # cosine lr schedule
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args["epochs"]))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Trainer:
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        model: torch.nn.Module,
        epochs: int = 10,
        learning_rate: float = 0.06,
        weight_decay: float = 5e-4,
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
        self.criterion = SimSiamLoss()
        self.validation_interval = validation_interval

    def train_epoch(self) -> None:
        self.model.train()
        # for aug1, aug2, _, _ in tqdm.tqdm(self.train_dataloader):
        for i, (images, _) in enumerate(self.train_dataloader):

            # aug1 = aug1.to(self.device)
            # aug2 = aug2.to(self.device)
            images[0] = images[0].cuda()
            images[1] = images[1].cuda()
            model_outputs = self.model(im_aug1=images[0], im_aug2=images[1])
            loss = self.criterion(
                model_outputs["z1"],
                model_outputs["z2"],
                model_outputs["p1"],
                model_outputs["p2"],
            )

            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss = loss.detach().cpu().item()
            self.avg_train_loss.update(loss)

    def train(self) -> None:
        for epoch in tqdm.trange(self.epochs):
            adjust_learning_rate(
                self.optimizer,
                epoch=epoch,
                args={"learning_rate": self.learning_rate, "epochs": self.epochs},
            )
            self.avg_train_loss.reset()
            self.train_epoch()
            avg_train_loss = self.avg_train_loss.compute()
            print(f"Epoch: {epoch}")
            print(f"Average train loss: {avg_train_loss}")
            if epoch != 0 and (epoch % self.validation_interval) == 0:
                val_acc = self.validation()
                print(f"Validation accuracy: {val_acc}")

    def validation(self) -> float:
        self.model.eval()
        train_features = []
        train_labels = []
        val_features = []
        val_labels = []

        with torch.no_grad():
            for batch in self.train_dataloader:
                images, label = batch
                img = images[-1]
                img = img.cuda()
                train_features.append(self.model.backbone(img).cpu())
                train_labels.append(label.cpu())

            train_features = torch.concat(train_features, dim=0).numpy()
            train_labels = torch.concat(train_labels, dim=0).numpy()

            for batch in self.val_dataloader:
                img, label = batch
                img = img.cuda()
                val_features.append(self.model.backbone(img).cpu())
                val_labels.append(label.cpu())

            val_features = torch.concat(val_features, dim=0).numpy()
            val_labels = torch.concat(val_labels, dim=0).numpy()

            knn = KNN(n_classes=10, top_k=[1], knn_k=200)
            val_acc = knn.knn_acc(
                val_features, val_labels, train_features, train_labels
            )

        return list(val_acc.values())[0]


class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()
        self.backbone = SimSiam.get_backbone("resnet18")
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        self.projector = projection_MLP(out_dim, 2048, 2)

        self.encoder = nn.Sequential(self.backbone, self.projector)

        self.predictor = prediction_MLP(2048)

    @staticmethod
    def get_backbone(backbone_name):
        return {
            "resnet18": resnet.ResNet18(),
            "resnet34": resnet.ResNet34(),
            "resnet50": resnet.ResNet50(),
            "resnet101": resnet.ResNet101(),
            "resnet152": resnet.ResNet152(),
        }[backbone_name]

    def forward(self, im_aug1, im_aug2):

        z1 = self.encoder(im_aug1)
        z2 = self.encoder(im_aug2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return {"z1": z1, "z2": z2, "p1": p1, "p2": p2}

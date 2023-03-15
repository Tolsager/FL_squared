from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from src.models.metrics import KNN


class SimSiam:
    pass


class SimSiamModel(SimSiam):
    def __init__(self, **kwargs):
        pass


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


class OurSimSiam(LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        projector: torch.nn.Module,
        predictor: torch.nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        n_classes: int = 10,
        top_k: list[int] = [1],
        knn_k: int = 10,
    ):
        super().__init__()
        # self.feature_dim = feature_dim
        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor
        self.criterion = torch.nn.CosineSimilarity(dim=1)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.n_classes = n_classes
        self.top_k = top_k
        self.knn_k = knn_k

    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z1 = self.projector(z1)
        z2 = self.backbone(x2)
        z2 = self.projector(z2)

        prediction1 = self.predictor(z1)
        prediction2 = self.predictor(z2)

        return prediction1, prediction2, z1.detach(), z2.detach()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        im1, im2, image, label = batch
        prediction1, prediction2, z1, z2 = self(im1, im2)
        loss = -0.5 * (
            self.criterion(prediction1, z2).mean()
            + self.criterion(prediction2, z1).mean()
        )
        train_features = self.backbone(image)
        self.log("train_loss", loss, on_epoch=True)
        return {"loss": loss, "train_features": train_features, "label": label}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        im, label = batch
        val_features = self.backbone(im)
        return {"val_features": val_features, "label": label}

    def training_epoch_end(self, training_step_outputs: list[tuple]):
        train_features = [i["train_features"] for i in training_step_outputs]
        train_labels = [i["label"] for i in training_step_outputs]

        train_features = torch.concat(train_features, dim=0)
        train_labels = torch.concat(train_labels, dim=0)

        train_features = train_features.detach().cpu().numpy()
        train_labels = train_labels.detach().cpu().numpy()

        knn = KNN(n_classes=self.n_classes, top_k=self.top_k, knn_k=self.knn_k)
        val_acc = knn.knn_acc(
            train_features, train_labels, self.val_features, self.val_labels
        )

        self.log("val_acc", val_acc)

        # self.lr.fit(predictions, labels)

    def validation_epoch_end(self, validation_step_outputs: list[tuple]):
        val_features = [i["val_features"] for i in validation_step_outputs]
        labels = [i["label"] for i in validation_step_outputs]

        val_features = torch.concat(val_features, dim=0)
        labels = torch.concat(labels, dim=0)

        val_features = val_features.cpu().numpy()
        labels = labels.cpu().numpy()

        self.val_features = val_features
        self.val_labels = labels

    def configure_optimizers(self):
        # return torch.optim.AdamW(
        #     self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        # )
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9,
        )
        return optimizer


def get_simsiam_predictor(embedding_dim: int = 432, hidden_dim: int = 200):
    predictor = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, hidden_dim, bias=False),
        torch.nn.BatchNorm1d(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, embedding_dim),
    )
    return predictor

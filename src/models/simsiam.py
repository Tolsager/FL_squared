from typing import Any, Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch

from pytorch_lightning import LightningModule
from pl_bolts.models.self_supervised.simsiam.simsiam_module import SimSiam


class SimSiamModel(SimSiam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _shared_step(self, batch: Any, batch_idx: int, step: str) -> torch.Tensor:
        """Shared evaluation step for training and validation loops."""
        imgs = batch
        img1, img2 = imgs[:2]

        # Calculate similarity loss in each direction
        loss_12 = self.calculate_loss(img1, img2)
        loss_21 = self.calculate_loss(img2, img1)

        # Calculate total loss
        total_loss = loss_12 + loss_21

        # Log loss
        if step == "train":
            self.log({"train_loss_12": loss_12, "train_loss_21": loss_21, "train_loss": total_loss})
        elif step == "val":
            self.log({"val_loss_12": loss_12, "val_loss_21": loss_21, "val_loss": total_loss})
        else:
            raise ValueError(f"Step '{step}' is invalid. Must be 'train' or 'val'.")

        return total_loss


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
        predictor: torch.nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        # self.feature_dim = feature_dim
        self.backbone = backbone
        self.predictor = predictor
        self.criterion = torch.nn.CosineSimilarity(dim=1)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)

        prediction1 = self.predictor(z1)
        prediction2 = self.predictor(z2)

        return prediction1, prediction2, z1.detach(), z2.detach()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        im1, im2, label = batch
        prediction1, prediction2, z1, z2 = self(im1, im2)
        loss = -0.5 * (
            self.criterion(prediction1, z2).mean()
            + self.criterion(prediction2, z1).mean()
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        im1, im2, label = batch
        prediction1, prediction2, z1, z2 = self(im1, im2)
        loss = -0.5 * (
            self.criterion(prediction1, z2).mean()
            + self.criterion(prediction2, z1).mean()
        )
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


def get_simsiam_predictor(embedding_dim: int = 432, hidden_dim: int = 200):
    predictor = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, embedding_dim),
    )
    return predictor

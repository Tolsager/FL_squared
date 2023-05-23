import copy
from datetime import datetime

import torch
import torchmetrics
import tqdm

import wandb


class SupervisedTrainer:
    def __init__(
        self,
        client_dataloaders: list[torch.utils.data.DataLoader],
        val_dataloader: torch.utils.data.Dataset,
        server_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        rounds: int,
        epochs: int,
        device: str = "cuda",
        n_classes: int = 10,
        learning_rate: float = 0.005,
        iid: bool = False,
    ):
        self.client_dataloaders = client_dataloaders
        self.val_dataloader = val_dataloader
        self.server_model = server_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.rounds = rounds
        self.epochs = epochs
        self.device = device
        self.n_clients = len(client_dataloaders)
        self.n_classes = n_classes
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_classes
        ).to(device)
        self.train_loss = torchmetrics.MeanMetric()
        self.learning_rate = learning_rate
        self.iid = iid
        self.models = [None for i in range(self.n_clients)]
        self.best_val_acc = 0.0
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

    @property
    def neutral_state_dict(self):
        state_dict = self.server_model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = torch.zeros_like(state_dict[key])
        return state_dict

    def train(self) -> torch.nn.Module:
        torch.cuda.empty_cache()
        for round in tqdm.trange(self.rounds):
            self.train_round()

            avg_state_dict = self.neutral_state_dict
            with torch.no_grad():
                # average client models to get the server model
                for client_idx in range(self.n_clients):
                    for key in avg_state_dict.keys():
                        params = self.models[client_idx].state_dict()[key]
                        avg_state_dict[key] += params

                for key in avg_state_dict.keys():
                    avg_state_dict[key] = avg_state_dict[key] / self.n_clients

            self.server_model.load_state_dict(avg_state_dict)

            # evaluate server model
            self.validate(self.server_model)
        return self.server_model

    def train_round(self):
        for client in tqdm.trange(self.n_clients):
            train_dataloader = self.client_dataloaders[client]
            model = copy.deepcopy(self.server_model)
            model.to(self.device)
            model.train()
            optimizer = self.optimizer(
                model.parameters(), lr=self.learning_rate, weight_decay=0
            )
            self.train_loss.reset()
            for epoch in range(self.epochs):
                for image, label in train_dataloader:
                    image = image.to(self.device)

                    logits = model(image)
                    # del image
                    label = label.to(self.device)
                    loss = self.criterion(logits, label)
                    # del logits
                    # del label

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if epoch == self.epochs - 1:
                        loss = float(loss)
                        self.train_loss(loss)

            avg_loss = self.train_loss.compute()
            wandb.log({f"client{client}_train_loss": avg_loss}, commit=False)
            model.to("cpu")
            self.models[client] = model

    def validate(self, model: torch.nn.Module):
        model.to(self.device)
        model.eval()
        self.val_acc.reset()
        with torch.no_grad():
            for image, label in self.val_dataloader:
                image = image.to(self.device)

                logits = model(image)
                # del image
                out = torch.argmax(logits, dim=1)

                label = label.to(self.device)
                self.val_acc(out, label)
                # del out
                # del label

        val_acc = self.val_acc.compute()
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(
                model.state_dict(), f"{'iid_' if self.iid else ''}Federated_model_{self.timestamp}.pth"
            )
        print(f"val_acc: {val_acc}")
        wandb.log({"val_acc": val_acc})
        model.to("cpu")

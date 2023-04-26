import copy

import torch
import torchmetrics
import tqdm

import wandb


class SupervisedTrainer:
    def __init__(
        self,
        client_dataloaders: list[torch.utils.data.DataLoader],
        val_dataloader: torch.utils.data.Dataset,
        models: list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        rounds: int,
        epochs: int,
        device: str = "cuda",
        n_classes: int = 10,
        learning_rate: float = 0.005,
    ):
        self.client_dataloaders = client_dataloaders
        self.val_dataloader = val_dataloader
        self.models = models
        self.optimizer = optimizer
        self.criterion = criterion
        self.rounds = rounds
        self.epochs = epochs
        self.device = device
        self.n_clients = len(models)
        self.n_classes = n_classes
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_classes
        ).to(device)
        self.train_loss = torchmetrics.MeanMetric()
        self.softmax = torch.nn.Softmax(dim=1)
        self.learning_rate = learning_rate

    @property
    def neutral_state_dict(self):
        state_dict = self.models[0].state_dict()
        for key in state_dict.keys():
            state_dict[key] = torch.zeros_like(state_dict[key])
        return state_dict

    def train(self):
        server_model = copy.deepcopy(self.models[0])
        for round in tqdm.trange(self.rounds):
            # update the optimizers with the new parameters
            self.optimizers = [
                self.optimizer(
                    m.parameters(), learning_rate=self.learning_rate, weight_decay=1e-2
                )
                for m in self.models
            ]
            self.train_round()

            # average client models to get the server model
            avg_state_dict = self.neutral_state_dict
            for client_idx in range(self.n_clients):
                for key in avg_state_dict.keys():
                    params = self.models[client_idx].state_dict()[key]
                    avg_state_dict[key] = avg_state_dict[key] + params

            for key in avg_state_dict.keys():
                avg_state_dict[key] = avg_state_dict[key] / self.n_clients

            server_model.load_state_dict(avg_state_dict)
            self.models = [copy.deepcopy(server_model) for _ in range(self.n_clients)]

            # evaluate server model
            self.validate(server_model)

    def train_round(self):
        for client in tqdm.trange(self.n_clients):
            train_dataloader = self.client_dataloaders[client]
            model = self.models[client]
            model.to(self.device)
            model.train()
            optimizer = self.optimizers[client]
            self.train_loss.reset()
            for epoch in range(self.epochs):
                for image, label in train_dataloader:
                    image = image.to(self.device)
                    label = label.to(self.device)

                    logits = model(image)
                    loss = self.criterion(logits, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # average the loss of the last batch of each epoch for efficiency
                loss = float(loss)
                self.train_loss(loss)
            avg_loss = self.train_loss.compute()
            wandb.log({f"client{client}_train_loss": avg_loss}, commit=False)
            model.to("cpu")

    def validate(self, model: torch.nn.Module):
        model.to(self.device)
        model.eval()
        self.val_acc.reset()
        for image, label in self.val_dataloader:
            image = image.to(self.device)
            label = label.to(self.device)

            logits = model(image)

            # apply softmax
            out = self.softmax(logits)

            self.val_acc(out, label)

        val_acc = self.val_acc.compute()
        wandb.log({"val_acc": val_acc})

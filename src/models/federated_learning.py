import copy

import torch
import torchmetrics

import wandb


class SupervisedTrainer:
    def __init__(
        self,
        client_dataloaders: list[torch.utils.data.Dataloader],
        val_dataloader: torch.utils.data.Dataset,
        models: list[torch.nn.Module],
        optimizers: list[torch.optim.Optimizer],
        criterion: torch.nn.modules.loss._Loss,
        rounds: int,
        epochs: int,
        device: str = "cuda",
    ):
        self.client_dataloaders = client_dataloaders
        self.val_dataloader = val_dataloader
        self.models = models
        self.optimizers = optimizers
        self.criterion = criterion
        self.rounds = rounds
        self.epochs = epochs
        self.device = device
        self.n_clients = len(models)
        self.val_acc = torchmetrics.Accuracy(task="multiclass")

    def train(self):
        server_model = copy.deepcopy(self.models[0])
        for round in self.n_rounds:
            self.train_round()

            # average client models to get the server model
            server_state_dict = server_model.state_dict()
            client_state_dicts = [m.state_dict() for m in self.models]
            for key in server_state_dict:
                params = [client_state_dicts[idx][key] for idx in range(self.n_clients)]
                params = torch.concat(params, dim=0)
                server_state_dict[key] = torch.mean(params)

            server_model = server_model.load_state_dict(server_state_dict)
            self.models = [copy.deepcopy(server_model) for _ in range(self.n_clients)]

            # evaluate server model
            self.validate(server_model)

    def train_round(self):
        for client in range(self.n_clients):
            train_dataloader = self.client_dataloaders[client]
            model = self.models[client]
            optimizer = self.optimizers[client]
            for epoch in range(self.epochs):
                model.train()
                for image, label in train_dataloader:
                    image = image.to(self.device)
                    label = label.to(self.device)

                    logits = model(image)
                    loss = self.criterion(logits, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def validate(self, model: torch.nn.Module):
        model.eval()
        self.val_acc.reset()
        for image, label in self.val_dataloader:
            image = image.to(self.device)
            label = label.to(self.device)

            logits = model(image)

            # apply softmax
            out = torch.nn.Softmax()(logits)

            self.val_acc(out, label)

        val_acc = self.val_acc.compute()
        wandb.log("val_acc", val_acc)

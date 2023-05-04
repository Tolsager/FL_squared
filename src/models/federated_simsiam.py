import copy

import torch
import torchmetrics
import tqdm

import wandb
from src.models import metrics, simsiam


class FedAvgSimSiamTrainer:
    def __init__(
        self,
        client_dataloaders: list[torch.utils.data.DataLoader],
        val_dataloader: torch.utils.data.DataLoader,
        server_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rounds: int,
        epochs: int,
        device: str = "cuda",
        n_classes: int = 10,
        learning_rate: float = 0.005,
        validation_interval: int = 5,
    ):
        self.client_dataloaders = client_dataloaders
        self.val_dataloader = val_dataloader
        self.server_model = server_model
        self.optimizer = optimizer
        self.criterion = simsiam.SimSiamLoss()
        self.rounds = rounds
        self.epochs = epochs
        self.device = device
        self.n_clients = len(client_dataloaders)
        self.n_classes = n_classes
        self.train_loss = torchmetrics.MeanMetric()
        self.learning_rate = learning_rate
        self.models = [None for i in range(self.n_clients)]
        self.validation_interval = validation_interval

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
            if self.val_dataloader is not None and (
                self.validation_interval == 1
                or (round != 0 and (round % self.validation_interval) == 0)
            ):
                val_acc = self.validate()
                wandb.log({"val_acc": val_acc})
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
                for aug1, aug2, _, _ in train_dataloader:
                    aug1 = aug1.to(self.device)
                    aug2 = aug2.to(self.device)
                    model_outputs = model(aug1, aug2)
                    loss = self.criterion(
                        model_outputs["z1"],
                        model_outputs["z2"],
                        model_outputs["p1"],
                        model_outputs["p2"],
                    )

                    # optimization
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

    def validate(self):
        self.server_model.eval()
        self.server_model.to(self.device)
        train_features = []
        train_labels = []
        val_features = []
        val_labels = []

        with torch.no_grad():
            for train_dataloader in self.client_dataloaders:
                for batch in train_dataloader:
                    _, _, img, label = batch
                    img = img.to(self.device)
                    train_features.append(self.server_model.backbone(img).cpu())
                    train_labels.append(label.cpu())

            train_features = torch.concat(train_features, dim=0).numpy()
            train_labels = torch.concat(train_labels, dim=0).numpy()

            for batch in self.val_dataloader:
                img, label = batch
                img = img.cuda()
                val_features.append(self.server_model.backbone(img).cpu())
                val_labels.append(label.cpu())
            self.server_model.to("cpu")

            val_features = torch.concat(val_features, dim=0).numpy()
            val_labels = torch.concat(val_labels, dim=0).numpy()

            knn = metrics.KNN(n_classes=10, top_k=[1], knn_k=200)
            val_acc = knn.knn_acc(
                val_features, val_labels, train_features, train_labels
            )

        return list(val_acc.values())[0]


class FedAvgSimSiamFinetuningTrainer:
    def __init__(
        self,
        client_dataloaders: list[torch.utils.data.DataLoader],
        supervised_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        server_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rounds: int,
        local_epochs: int,
        finetuning_epochs: int,
        device: str = "cuda",
        n_classes: int = 10,
        learning_rate: float = 0.005,
        validation_interval: int = 5,
        supervised_learning_rate: float = 0.01,
    ):
        self.client_dataloaders = client_dataloaders
        self.val_dataloader = val_dataloader
        self.supervised_dataloader = supervised_dataloader
        self.server_model = server_model
        self.optimizer = optimizer
        self.finetuning_epochs = finetuning_epochs
        self.criterion = simsiam.SimSiamLoss()
        self.supervised_criterion = torch.nn.CrossEntropyLoss()
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.device = device
        self.n_clients = len(client_dataloaders)
        self.n_classes = n_classes
        self.train_loss = torchmetrics.MeanMetric()
        self.learning_rate = learning_rate
        self.models = [None for i in range(self.n_clients)]
        self.validation_interval = validation_interval
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=10).to(device)
        self.supervised_learning_rate = supervised_learning_rate

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
            wandb.log({})

            # evaluate server model
        self.server_model.to(self.device)
        optimizer = self.optimizer(
            self.server_model.parameters(),
            lr=self.supervised_learning_rate,
            weight_decay=0,
        )
        for epoch in tqdm.trange(self.finetuning_epochs):
            self.train_supervised(optimizer)
            self.validate()
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
            for epoch in range(self.local_epochs):
                for aug1, aug2, _, _ in train_dataloader:
                    aug1 = aug1.to(self.device)
                    aug2 = aug2.to(self.device)
                    model_outputs = model(aug1, aug2)
                    del aug1
                    del aug2
                    loss = self.criterion(
                        model_outputs["z1"],
                        model_outputs["z2"],
                        model_outputs["p1"],
                        model_outputs["p2"],
                    )
                    del model_outputs

                    # optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if epoch == self.local_epochs - 1:
                        loss = float(loss)
                        self.train_loss(loss)

            avg_loss = self.train_loss.compute()
            wandb.log({f"client{client}_train_loss": avg_loss}, commit=False)
            model.to("cpu")
            self.models[client] = model

    def validate(self):
        self.server_model.eval()
        self.server_model.to(self.device)

        self.val_acc.reset()
        for image, label in self.val_dataloader:
            image = image.to(self.device)

            logits = self.server_model(image)
            del image
            out = torch.argmax(logits, dim=1)

            label = label.to(self.device)
            self.val_acc(out, label)
            del out
            del label

        val_acc = self.val_acc.compute()
        print(f"val_acc: {val_acc}")
        wandb.log({"val_acc": val_acc})

    def train_supervised(self, optimizer):
        self.server_model.train()

        self.train_loss.reset()

        for image, label in self.supervised_dataloader:
            image = image.to(self.device)

            logits = self.server_model(image)
            label = label.to(self.device)
            loss = self.supervised_criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = float(loss)
            self.train_loss.update(loss)

        train_loss = self.train_loss.compute()
        wandb.log({"supervised_train_loss": train_loss})

import torch

class Trainer:
    def __init__(self, client_dataloaders: list[torch.utils.data.Dataloader], val_dataloader: torch.utils.data.Dataset, models: list[torch.nn.Module], optimizers: list[torch.optim.Optimizer], criterion: torch.nn.modules.loss._Loss, rounds: int, epochs: int, device: str="cuda"):
        self.client_dataloaders = client_dataloaders
        self.val_dataloader = val_dataloader
        self.models = models
        self.optimizers = optimizers
        self.criterion = criterion
        self.rounds = rounds
        self.epochs = epochs
        self.device = device
        self.n_clients = len(models)

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
                    

                    


        



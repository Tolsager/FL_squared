from src.models import federated_learning as fr
from src.data import process_data, make_dataset
import torch
import torchvision
from src.models.res

train_ds, test_ds = make_dataset.load_dataset()
train_ds = process_data.AugmentedDataset(train_ds, torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS))


def test_supervised_trainer():
    train_subset = torch.utils.data.Subset(train_ds, [*range(4)])
    val_subset = torch.utils.data.Subset(train_ds, [*range(2)])
    train_dl = torch.utils.data.DataLoader(train_subset, batch_size=2)
    val_dl = torch.utils.data.DataLoader(val_subset, batch_size=2)


    client_dataloaders = [
        torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        for ds in train_datasets
    ]



    device = "cpu"
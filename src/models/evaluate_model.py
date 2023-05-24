import os
from typing import Tuple

import torch
import torchmetrics
import torchvision

from src.data import make_dataset, process_data
from src.models import metrics, resnet, simsiam


class EvalModel:
    def __init__(self, model_weights: str, model: torch.nn.Module):
        self.model_weights = model_weights
        self.model = model
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to("cuda")

    def prepare_model(self) -> None:
        weights_path = os.path.join("models", self.model_weights)
        self.model.cuda()
        self.model.load_state_dict(torch.load(weights_path), strict=True)
        self.model.eval()

    def get_dl(self) -> torch.utils.data.DataLoader:
        _, test_ds = make_dataset.load_dataset()
        test_ds = process_data.AugmentedDataset(
            test_ds,
            torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS),
        )
        return torch.utils.data.DataLoader(test_ds, batch_size=400)

    def eval(self) -> float:
        self.prepare_model()
        test_dl = self.get_dl()
        with torch.no_grad():
            for batch in test_dl:
                pred, label = self.predict(batch)
                self.acc(pred, label)

        return self.acc.compute()

    def predict(self, batch) -> Tuple[torch.Tensor]:
        raise NotImplementedError


class EvalSupervised(EvalModel):
    def __init__(self, *args):
        super().__init__(*args)

    def predict(self, batch):
        image, label = batch
        image = image.to("cuda")
        label = label.to("cuda")

        out = self.model(image)
        pred = torch.argmax(out, dim=1)
        return pred, label


class EvalSimSiam(EvalModel):
    def __init__(self, *args):
        super().__init__(*args)

    def get_dls(self):
        train_ds, test_ds = make_dataset.load_dataset()
        test_ds = process_data.AugmentedDataset(
            test_ds,
            torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS),
        )
        train_ds = process_data.AugmentedDataset(
            train_ds,
            torchvision.transforms.Compose(process_data.CIFAR10_STANDARD_TRANSFORMS),
        )
        return torch.utils.data.DataLoader(
            train_ds, batch_size=400
        ), torch.utils.data.DataLoader(test_ds, batch_size=400)

    def eval(self):
        self.prepare_model()
        self.train_dataloader, self.val_dataloader = self.get_dls()
        train_features = []
        train_labels = []
        val_features = []
        val_labels = []

        with torch.no_grad():
            for batch in self.train_dataloader:
                img, label = batch
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

            knn = metrics.KNN(n_classes=10, top_k=[1], knn_k=200)
            val_acc = knn.knn_acc(
                val_features, val_labels, train_features, train_labels
            )

        return list(val_acc.values())[0]


def write_results(exp_name: str, acc: float):
    with open("reports/test_accuracies.txt", "a+") as f:
        f.write(f"{exp_name}: {acc}\n")


if __name__ == "__main__":
    # model = resnet.ResNet18Classifier(10)
    # evaluator = EvalSupervised("Centralized.pth", model)
    # acc = evaluator.eval()
    # write_results("Centralized", acc)
    #
    # model = resnet.ResNet18Classifier(10)
    # evaluator = EvalSupervised("Federated.pth", model)
    # acc = evaluator.eval()
    # write_results("Federated", acc)
    #
    # model = resnet.ResNet18Classifier(10)
    # evaluator = EvalSupervised("Federated_non-iid.pth", model)
    # acc = evaluator.eval()
    # write_results("Federated non-i.i.d.", acc)
    #
    # model = simsiam.SimSiam()
    # evaluator = EvalSimSiam("SimSiam.pth", model)
    # acc = evaluator.eval()
    # write_results("SimSiam", acc)

    model = simsiam.SimSiam()
    evaluator = EvalSimSiam("FLS.pth", model)
    acc = evaluator.eval()
    write_results("FLS", acc)

    model = simsiam.SimSiam()
    evaluator = EvalSimSiam("FLS_non-iid.pth", model)
    acc = evaluator.eval()
    write_results("FLS non-i.i.d.", acc)


    model = simsiam.SimSiam(n_classes=10)
    evaluator = EvalSupervised("iid_Finetuned_FLS_2023_05_23_14_13.pth", model)
    acc = evaluator.eval()
    write_results("iid_Finetuned_FLS_2023_05_23_14_13", acc)

    model = simsiam.SimSiam(n_classes=10)
    evaluator = EvalSupervised("unfrozen_iid_Finetuned_FLS_2023_05_23_14_50.pth", model)
    acc = evaluator.eval()
    write_results("unfrozen_iid_Finetuned_FLS_2023_05_23_14_50", acc)

    model = simsiam.SimSiam(n_classes=10)
    evaluator = EvalSupervised("non_iid_Finetuned_FLS_2023_05_23_14_52.pth", model)
    acc = evaluator.eval()
    write_results("non_iid_Finetuned_FLS_2023_05_23_14_52", acc)

    model = simsiam.SimSiam(n_classes=10)
    evaluator = EvalSupervised("unfrozen_non_iid_Finetuned_FLS_2023_05_23_15_23.pth", model)
    acc = evaluator.eval()
    write_results("unfrozen_non_iid_Finetuned_FLS_2023_05_23_15_23", acc)
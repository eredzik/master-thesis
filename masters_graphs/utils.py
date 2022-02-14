import gc
import torch
from GPUtil import showUtilization as gpu_usage


# def free_gpu_cache():
#     print("Initial GPU Usage")
#     gpu_usage()

#     gc.collect()
#     torch.cuda.empty_cache()

#     # cuda.select_device(0)
#     # cuda.close()
#     # cuda.select_device(0)

#     print("GPU Usage after emptying the cache")
#     gpu_usage()


# free_gpu_cache()
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import auc, roc_curve
from torch_geometric.datasets import CitationFull
from torch_geometric.transforms import (
    NormalizeFeatures,
    RandomNodeSplit,
)
from sklearn.metrics import f1_score
from torch_geometric.datasets import (
    CitationFull,
    Reddit,
)


def load_dataset(name, device):
    if name == "Citeseer":
        dataset = CitationFull(
            root="data/CitationFull",
            name="Citeseer",
            transform=NormalizeFeatures(),
        )
        data = dataset[0]
        RandomNodeSplit(
            split="test_rest",
            num_train_per_class=30,
        )(data)
        data.to(device, "x", "y", "edge_index")
        return data, dataset
    elif name == "Pubmed":
        dataset = CitationFull(
            root="data/CitationFull",
            name="Pubmed",
        )
        data = dataset[0]
        RandomNodeSplit(split="test_rest")(data)
        data.to(device, "x", "y", "edge_index")
        return data, dataset
    elif name == "Reddit":
        dataset = Reddit(root="data/Reddit")
        data = dataset[0]
        return data, dataset
    else:
        raise Exception("wrong dataset name")


def print_dataset_summary(
    dataset, data, dataset_name
):
    print()
    print(f"Dataset: {dataset}:")
    print("======================")
    print(f"Number of graphs: {len(dataset)}")
    print(
        f"Number of features: {dataset.num_features}"
    )
    print(
        f"Number of classes: {dataset.num_classes}"
    )
    counts = []
    for cls in range(dataset.num_classes):
        class_count = (data.y == cls).sum()
        print(
            "Number of class ", cls, class_count
        )
        counts.append(class_count)

    fig, ax = plt.subplots()

    ax.bar(range(dataset.num_classes), counts)
    if dataset_name == "Reddit":
        ax.set_xticks(
            range(dataset.num_classes),
            [
                x if x % 2 == 0 else None
                for x in range(
                    dataset.num_classes
                )
            ],
        )
    else:
        ax.set_xticks(range(dataset.num_classes))
    ax.set_title(
        f"Rozkład klas dla zbioru danych {dataset_name}"
    )
    ax.set_xlabel("Klasa")
    ax.set_ylabel("Liczebność")
    plt.gcf().set_dpi(150)

    plt.show()

    print()
    print(data)
    print(
        "==========================================================================================================="
    )

    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(
        f"Average node degree: {data.num_edges / data.num_nodes:.2f}"
    )
    print(
        f"Number of training nodes: {data.train_mask.sum()}"
    )
    print(
        f"Number of validation nodes: {data.val_mask.sum()}"
    )
    print(
        f"Number of testing nodes: {data.test_mask.sum()}"
    )
    print(
        f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}"
    )

    print(
        f"Has isolated nodes: {data.has_isolated_nodes()}"
    )
    print(
        f"Has self-loops: {data.has_self_loops()}"
    )
    print(
        f"Is undirected: {data.is_undirected()}"
    )


class Plots:
    def __init__(self):
        self.fig, (
            self.ax1,
            self.ax2,
            self.ax3,
            self.ax4,
        ) = plt.subplots(4)

    def plot_plots(
        self, losses, accuracies, f1_scores, gini
    ):

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.fig.set_size_inches(
            10.5, 10.5, forward=True
        )

        self.ax1.plot(
            losses["train"],
            label="Zbiór treningowy",
        )
        self.ax1.plot(
            losses["valid"],
            label="Zbiór walidacyjny",
        )
        self.ax1.plot(
            losses["test"], label="Zbiór testowy"
        )
        self.ax1.set_ylabel(
            "Wartość funkcji kosztu"
        )

        self.ax2.plot(
            accuracies["train"],
            label="Zbiór treningowy",
        )
        self.ax2.plot(
            accuracies["valid"],
            label="Zbiór walidacyjny",
        )
        self.ax2.plot(
            accuracies["test"],
            label="Zbiór testowy",
        )
        self.ax2.set_ylabel("Wartość celności")

        self.ax3.plot(
            f1_scores["train"],
            label="Zbiór treningowy",
        )
        self.ax3.plot(
            f1_scores["valid"],
            label="Zbiór walidacyjny",
        )
        self.ax3.plot(
            f1_scores["test"],
            label="Zbiór testowy",
        )

        self.ax4.plot(
            gini["train"],
            label="Zbiór treningowy",
        )
        self.ax4.plot(
            gini["valid"],
            label="Zbiór walidacyjny",
        )
        self.ax4.plot(
            gini["test"], label="Zbiór testowy"
        )
        self.ax4.set_ylabel(
            "Wartość miary Giniego"
        )

        self.ax4.set_ylabel("Wartość miary F1")
        self.ax4.set_xlabel("Epoch")
        self.ax4.legend()


class EarlyStopperNaive:
    def __init__(
        self,
        patience_epochs=5,
        minimal_reduction=0.001,
        min_number_of_epochs=10,
    ):
        self.patience_epochs = patience_epochs
        self.current_patience = 0
        self.min_loss = None
        self.minimal_reduction = minimal_reduction
        self.min_number_of_epochs = (
            min_number_of_epochs
        )
        self.current_epoch = 0

    def __call__(self, loss):
        should_break = False
        self.current_epoch += 1
        if self.min_loss is None:
            self.min_loss = loss

        if (
            self.min_loss
            * (1 - self.minimal_reduction)
            < loss
        ):
            self.current_patience += 1
        else:
            self.current_patience = 0

        self.min_loss = min(loss, self.min_loss)
        if (
            self.current_patience
            > self.patience_epochs
        ):
            should_break = True
        if (
            self.current_epoch
            < self.min_number_of_epochs
        ):
            should_break = False
        return should_break, self.current_patience


class MetricStore:
    def __init__(self, masks, num_classes):
        self.masks = masks
        self.num_classes = num_classes
        self.store = {
            "accuracy": {
                "train": [],
                "test": [],
                "valid": [],
            },
            "f1_score": {
                "train": [],
                "test": [],
                "valid": [],
            },
            "gini": {
                "train": [],
                "test": [],
                "valid": [],
            },
            "fpr": {
                "train": [],
                "test": [],
                "valid": [],
            },
            "tpr": {
                "train": [],
                "test": [],
                "valid": [],
            },
            "loss": {
                "train": [],
                "test": [],
                "valid": [],
            },
        }

    def calculate_all(self, model_pred, y):
        self.calculate_accuracies(model_pred, y)
        self.calculate_f1_score(model_pred, y)
        self.calculate_roc(model_pred, y)

    def calculate_loss(
        self, model_pred, y, criterion
    ):
        for ds_type in self.masks.keys():
            self.store["loss"][ds_type].append(
                criterion(
                    model_pred[
                        self.masks[ds_type]
                    ],
                    y[self.masks[ds_type]],
                )
            )

    def calculate_accuracies(self, model_pred, y):
        pred = model_pred.argmax(1)
        for ds_type in self.masks.keys():
            correct_predicted = (
                pred[self.masks[ds_type]]
                == y[self.masks[ds_type]]
            )
            nominator = int(
                correct_predicted.sum()
            )
            denominator = int(
                self.masks[ds_type].sum()
            )
            accuracy = nominator / denominator
            self.store["accuracy"][
                ds_type
            ].append(accuracy)

    def calculate_f1_score(self, model_pred, y):
        pred = model_pred.argmax(1)
        for ds_type in self.masks.keys():
            res = f1_score(
                y[self.masks[ds_type]],
                pred[self.masks[ds_type]],
                average="weighted",
            )
            self.store["f1_score"][
                ds_type
            ].append(res)

    def calculate_roc(self, model_pred, y):
        for ds_type in self.masks.keys():
            (
                roc_auc,
                fpr,
                tpr,
            ) = self.get_roc_auc_data(
                model_pred[self.masks[ds_type]],
                y[self.masks[ds_type]],
                self.num_classes,
            )
            self.store["gini"][ds_type].append(
                roc_auc["micro"] * 2 - 1
            )
            self.store["fpr"][ds_type].append(fpr)
            self.store["tpr"][ds_type].append(tpr)

    def plot_plots(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4
        )
        fig.set_size_inches(
            10.5, 10.5, forward=True
        )

        ax1.plot(
            self.store["loss"]["train"],
            label="Zbiór treningowy",
        )
        ax1.plot(
            self.store["loss"]["valid"],
            label="Zbiór walidacyjny",
        )
        ax1.plot(
            self.store["loss"]["test"],
            label="Zbiór testowy",
        )
        ax1.set_ylabel("Wartość funkcji kosztu")

        ax2.plot(
            self.store["accuracy"]["train"],
            label="Zbiór treningowy",
        )
        ax2.plot(
            self.store["accuracy"]["valid"],
            label="Zbiór walidacyjny",
        )
        ax2.plot(
            self.store["accuracy"]["test"],
            label="Zbiór testowy",
        )
        ax2.set_ylabel("Wartość trafności")

        ax3.plot(
            self.store["f1_score"]["train"],
            label="Zbiór treningowy",
        )
        ax3.plot(
            self.store["f1_score"]["valid"],
            label="Zbiór walidacyjny",
        )
        ax3.plot(
            self.store["f1_score"]["test"],
            label="Zbiór testowy",
        )
        ax3.set_ylabel("Wartość miary F1")

        ax4.plot(
            self.store["gini"]["train"],
            label="Zbiór treningowy",
        )
        ax4.plot(
            self.store["gini"]["valid"],
            label="Zbiór walidacyjny",
        )
        ax4.plot(
            self.store["gini"]["test"],
            label="Zbiór testowy",
        )
        ax4.set_ylabel("Wartość miary Giniego")
        ax4.set_xlabel("Epoch")
        ax4.legend()

    @staticmethod
    def get_roc_auc_data(
        y_score, y_true, n_classes
    ):

        fpr = {}
        tpr = {}
        roc_auc = {}
        b = np.zeros((y_true.shape[0], n_classes))

        for i in range(n_classes):
            b[y_true == i, i] = 1
            fpr[i], tpr[i], _ = roc_curve(
                b[:, i], y_score[:, i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(
            b.ravel(), y_score.ravel()
        )
        roc_auc["micro"] = auc(
            fpr["micro"], tpr["micro"]
        )

        all_fpr = np.unique(
            np.concatenate(
                [fpr[i] for i in range(n_classes)]
            )
        )

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(
                all_fpr, fpr[i], tpr[i]
            )

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(
            fpr["macro"], tpr["macro"]
        )
        return roc_auc, fpr, tpr


def plot_roc_auc(data_dict, outdir=None):

    lw = 2

    fig = plt.figure()
    colors = cycle(
        [
            "aqua",
            "darkorange",
            "cornflowerblue",
            "deeppink",
        ]
    )
    fig.set_size_inches(6, 6, forward=True)
    for (model, vals), color in zip(
        data_dict.items(), colors
    ):

        plt.plot(
            vals["fpr"]["micro"],
            vals["tpr"]["micro"],
            label=f"{model}",
            color=color,
            linewidth=2,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        lw=lw,
        label="Model losowy",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1-Specyficzność")
    plt.ylabel("Czułość")

    plt.title(
        "Krzywe ROC dla poszczególnych modeli"
    )
    plt.legend(loc="lower right")

    plt.gcf().set_dpi(150)
    if outdir:
        plt.savefig(outdir)
    else:
        plt.show()

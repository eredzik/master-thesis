import pickle

from matplotlib.pyplot import plot
from utils import MetricStore, plot_roc_auc
import os

dataset_to_load, outdir = os.environ["dataset_name"], os.environ["outdir_name"]
stores = {}
for store_to_load in [
    "ann_regression",
    "dummy_regression_result",
    "gnn_regression",
    "logistic_regression_result",
]:
    with open(f"masters_graphs/{outdir}/{store_to_load}.pickle", "rb") as f:
        stores[store_to_load] = pickle.load(f)


def get_from_store(store):
    return {
        "tpr": store.store["tpr"]["test"][-1],
        "fpr": store.store["fpr"]["test"][-1],
    }


plot_roc_auc(
    {
        # "Model losowy": get_from_store(stores["dummy_regression_result"]),
        "Regresja logistyczna": get_from_store(stores["logistic_regression_result"]),
        "Głęboka sieć neuronowa": get_from_store(stores["ann_regression"]),
        "Grafowa sieć neuronowa": get_from_store(stores["gnn_regression"]),
    },
    outdir=f"masters_graphs/{outdir}/{outdir}_roc_auc.png",
)

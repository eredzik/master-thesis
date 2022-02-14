from utils import MetricStore, load_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid

import os

device = "cpu"

dataset_to_load, outdir, = (
    os.environ["dataset_name"],
    os.environ["outdir_name"],
)

data, dataset = load_dataset(dataset_to_load, device)
masks = {
    "train": data.train_mask.to(device),
    "test": data.test_mask.to(device),
    "valid": data.val_mask.to(device),
}


def mlp_param_search():

    grid = ParameterGrid(
        {
            "n_layers": [1, 2, 3],
            "layer_size": [16, 128, 512],
            "activation": ["tanh", "relu"],
        }
    )

    with open(f"masters_graphs/{outdir}/results_gridsearch_ann", "w") as f:

        for param in grid:

            clf = MLPClassifier(
                hidden_layer_sizes=tuple([param["layer_size"] * param["n_layers"]]),
                activation=param["activation"],
            )

            clf.fit(data.x[data.train_mask].cpu(), data.y[data.train_mask].cpu())

            predictions = clf.predict_proba(data.x.cpu())
            mlp_store = MetricStore(masks=masks, num_classes=dataset.num_classes)
            mlp_store.calculate_all(predictions, data.y.cpu().detach().numpy())

            f.write(
                str(param["n_layers"])
                + "_"
                + str(param["layer_size"])
                + "_"
                + str(param["activation"])
                + "="
                + str(mlp_store.store["gini"]["valid"][-1].round(3))
                + "\n"
            )
            f.write(str(round(mlp_store.store["accuracy"]["valid"][-1], 3)) + "\n")
            f.write(str(mlp_store.store["f1_score"]["valid"][-1].round(3)) + "\n")
            f.write(str(mlp_store.store["gini"]["valid"][-1].round(3)) + "\n")


mlp_param_search()

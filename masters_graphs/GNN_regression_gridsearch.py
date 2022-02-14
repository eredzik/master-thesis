import pickle
from utils import (
    MetricStore,
    load_dataset,
    EarlyStopperNaive,
)

from sklearn.model_selection import ParameterGrid
from GNN_basic import GCN, train
import torch

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

(
    dataset_to_load,
    outdir,
    n_layers,
    n_neurons,
    activation,
) = (
    "Citeseer",
    "citeseer",
    2,
    128,
    "relu",
)
data, dataset = load_dataset(
    dataset_to_load, device
)
masks = {
    "train": data.train_mask.to(device),
    "test": data.test_mask.to(device),
    "valid": data.val_mask.to(device),
}


grid = ParameterGrid(
    {
        "n_layers": [1, 2, 3],
        "layer_size": [16, 128, 512],
        "activation": ["tanh", "relu"],
    }
)
# grid_citeseer = ParameterGrid({
#     "n_layers": [2],
#     "layer_size": [512],
#     "activation":['relu']})
# grid = grid_citeseer
with open(
    f"masters_graphs/{outdir}/results_gridsearch_gnn",
    "w",
) as f:
    for param in grid:

        model = GCN(
            hidden_channels=param["layer_size"],
            n_layers=param["n_layers"],
            activation=param["activation"],
            n_classes=dataset.num_classes,
            n_fetures=dataset.num_features,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
            weight_decay=5e-4,
        )
        criterion = torch.nn.CrossEntropyLoss()
        store = MetricStore(
            masks={
                k: v.cpu().detach().numpy()
                for (k, v) in masks.items()
            },
            num_classes=dataset.num_classes,
        )
        stopper = EarlyStopperNaive(
            minimal_reduction=0.000,
            patience_epochs=20,
        )

        for epoch in range(1, 101):
            train(
                data=data,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
            )
            # should_break, patience = stopper(-gcn_store.store['gini']['valid'][-1])
            # if should_break:
            # break

            # print(f'Epoch: {epoch:03d}, Loss train: ', float(gcn_store.store['loss']['train'][-1]), 'Loss valid: ', float(gcn_store.store['gini']['valid'][-1]),f' Patience: {patience}')
        out = model(data.x, data.edge_index)
        store.calculate_all(
            out.cpu().detach().numpy(),
            data.y.cpu().detach().numpy(),
        )

        f.write(
            str(param["n_layers"])
            + "_"
            + str(param["layer_size"])
            + "_"
            + str(param["activation"])
            + "="
            + str(
                store.store["gini"]["valid"][
                    -1
                ].round(3)
            )
            + "\n"
        )
        f.write(
            str(
                round(
                    store.store["accuracy"][
                        "valid"
                    ][-1],
                    3,
                )
            )
            + "\n"
        )
        f.write(
            str(
                store.store["f1_score"]["valid"][
                    -1
                ].round(3)
            )
            + "\n"
        )
        f.write(
            str(
                store.store["gini"]["valid"][
                    -1
                ].round(3)
            )
            + "\n"
        )
# gcn_store.calculate_loss(out, data.y, criterion)
# gcn_store.plot_plots()

# print(gcn_store.store['gini']['test'][-1].round(3))
# print(round(gcn_store.store['accuracy']['test'][-1], 3))
# print(gcn_store.store['f1_score']['test'][-1].round(3))

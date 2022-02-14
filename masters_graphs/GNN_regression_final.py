from GNN_basic import GCN, train
from utils import (
    load_dataset,
    MetricStore,
    EarlyStopperNaive,
)
import torch
import pickle
import os

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
    os.environ["dataset_name"],
    os.environ["outdir_name"],
    int(os.environ["gnn_n_layers"]),
    int(os.environ["gnn_n_neurons"]),
    os.environ["gnn_activation"],
)

data, dataset = load_dataset(
    dataset_to_load, device
)
masks = {
    "train": data.train_mask.to(device),
    "test": data.test_mask.to(device),
    "valid": data.val_mask.to(device),
}

model = GCN(
    hidden_channels=n_neurons,
    n_layers=n_layers,
    activation=activation,
    n_classes=dataset.num_classes,
    n_fetures=dataset.num_features,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01, weight_decay=5e-4
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
    minimal_reduction=0.000, patience_epochs=20
)

for epoch in range(1, 101):
    train(
        data=data,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
    )
out = model(data.x, data.edge_index)
store.calculate_all(
    out.cpu().detach().numpy(),
    data.y.cpu().detach().numpy(),
)

print(
    round(store.store["accuracy"]["test"][-1], 3)
)
print(
    store.store["f1_score"]["test"][-1].round(3)
)
print(store.store["gini"]["test"][-1].round(3))

with open(
    f"masters_graphs/{outdir}/gnn_regression.pickle",
    "wb",
) as f:
    pickle.dump(store, f)

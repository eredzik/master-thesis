from utils import (
    Plots,
    MetricStore,
    EarlyStopperNaive,
    load_dataset,
)

import os
import pickle
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
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
    os.environ["dataset_name"],
    os.environ["outdir_name"],
    int(os.environ["gnn_n_layers"]),
    int(os.environ["gnn_n_neurons"]),
    os.environ["gnn_activation"],
)

data, dataset = load_dataset(
    dataset_to_load, device
)
del (
    dataset.data.x,
    dataset.data.y,
    dataset.data.edge_index,
)
masks = {
    "train": data.train_mask,
    "test": data.test_mask,
    "valid": data.val_mask,
}


class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_features,
        num_classes,
        device,
    ):
        super().__init__()
        self.device = device
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(
                num_features, hidden_channels
            )
        )
        self.convs.append(
            SAGEConv(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(
            total=len(subgraph_loader.dataset)
            * len(self.convs),
            leave=False,
        )
        pbar.set_description(
            f"Epoch {epoch} valid"
        )

        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[
                    batch.n_id.to(x_all.device)
                ].to(self.device)
                x = conv(
                    x,
                    batch.edge_index.to(
                        self.device
                    ),
                )
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(
                    x[: batch.batch_size].cpu()
                )
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


kwargs = {
    "batch_size": 256,
    "num_workers": 2,
    "persistent_workers": True,
}
train_loader = NeighborLoader(
    data,
    input_nodes=data.train_mask,
    num_neighbors=[25, 10],
    shuffle=True,
    **kwargs,
)
valid_loader = NeighborLoader(
    data,
    input_nodes=data.val_mask,
    num_neighbors=[25, 10],
    shuffle=True,
    **kwargs,
)
test_loader = NeighborLoader(
    data,
    input_nodes=data.test_mask,
    num_neighbors=[25, 10],
    shuffle=True,
    **kwargs,
)
model = GraphSAGE(
    hidden_channels=128,
    num_classes=dataset.num_classes,
    num_features=dataset.num_features,
    device=device,
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01
)
criterion = torch.nn.CrossEntropyLoss()
gcn_store = MetricStore(
    masks=masks, num_classes=dataset.num_classes
)
mini_losses = []
mini_losses_labels = []


def train():

    model.train()

    pbar = tqdm(
        total=int(len(train_loader.dataset)),
        leave=False,
    )
    pbar.set_description(
        f"Epoch {epoch:02d} train"
    )

    total_loss = (
        total_correct
    ) = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[: batch.batch_size]
        out = model(
            batch.x, batch.edge_index.to(device)
        )[: batch.batch_size]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_correct += int(
            (out.argmax(dim=-1) == y).sum()
        )
        total_examples += batch.batch_size
        mini_losses.append(float(loss))
        mini_losses_labels.append(None)
        pbar.update(batch.batch_size)
    pbar.close()
    return total_loss


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(data.x, subgraph_loader)
    y = data.y.to(out.device)

    gcn_store.calculate_all(out, y)
    gcn_store.calculate_loss(out, y, criterion)
    mini_losses.append(
        gcn_store.store["loss"]["train"][-1]
    )
    mini_losses_labels.append(epoch)
    model.train()


plotter = Plots()
stopper = EarlyStopperNaive(
    minimal_reduction=0.009
)

for epoch in range(5):
    train_loss = train()
    test()
    valid_loss = gcn_store.store["loss"]["valid"][
        -1
    ]
    should_stop, patience = stopper(valid_loss)
    if should_stop:
        break
    plotter.plot_plots(
        gcn_store.store["loss"],
        gcn_store.store["accuracy"],
        gcn_store.store["f1_score"],
        gcn_store.store["gini"],
    )
    print(
        f"Epoch: {epoch:03d}, Loss train: {train_loss}, Loss valid: {valid_loss}, Patience: {patience}"
    )


print(
    round(
        gcn_store.store["accuracy"]["test"][-1], 3
    )
)
print(
    gcn_store.store["f1_score"]["test"][-1].round(
        3
    )
)
print(
    gcn_store.store["gini"]["test"][-1].round(3)
)
with open(
    f"masters_graphs/{outdir}/gnn_regression.pickle",
    "wb",
) as f:
    pickle.dump(gcn_store, f)

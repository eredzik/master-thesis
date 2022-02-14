from torch_geometric.nn import GCNConv
import torch


class GCN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_layers,
        activation,
        n_fetures,
        n_classes,
    ):
        super().__init__()
        torch.manual_seed(1234567)
        self.convs = torch.nn.ModuleList()

        self.activation = activation
        for i in range(n_layers):
            if n_layers == 1:
                self.convs.append(
                    GCNConv(n_fetures, n_classes)
                )
            elif i == 0:
                self.convs.append(
                    GCNConv(
                        n_fetures, hidden_channels
                    )
                )
            elif i == n_layers - 1:
                self.convs.append(
                    GCNConv(
                        hidden_channels, n_classes
                    )
                )
            else:
                self.convs.append(
                    GCNConv(
                        hidden_channels,
                        hidden_channels,
                    )
                )

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            if self.activation == "tanh":
                x = x.tanh()
            else:
                x = x.relu_()
        x = self.convs[-1](x, edge_index)
        return x


def train(model, optimizer, data, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask],
    )
    loss.backward()

    optimizer.step()
    model.eval()
    out = model(data.x, data.edge_index)

    model.train()

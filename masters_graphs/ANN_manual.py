import torch
from torch.nn import Linear
import torch.nn.functional as F

from utils import EarlyStopperNaive, MetricStore


class MLP(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_features,
        n_classes,
        n_layers,
    ):
        super().__init__()
        torch.manual_seed(12345)
        self.layers = torch.nn.ModuleList()

        for i in range(n_layers):
            if n_layers == 1:
                self.layers.append(
                    Linear(n_features, n_classes)
                )
            elif i == 0:
                self.layers.append(
                    Linear(
                        n_features,
                        hidden_channels,
                    )
                )
            elif i == n_layers - 1:
                self.layers.append(
                    Linear(
                        hidden_channels, n_classes
                    )
                )
            else:
                self.layers.append(
                    Linear(
                        hidden_channels,
                        hidden_channels,
                    )
                )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation == "tanh":
                x = x.tanh()
            else:
                x = x.relu_()
        x = self.convs[-1](x)
        return x


class TrainableMLP:
    def __init__(
        self,
        hidden_channels,
        n_features,
        n_classes,
        n_layers,
        device,
    ):
        self.model = MLP(
            hidden_channels,
            n_features,
            n_classes,
            n_layers,
        ).to(device)
        self.criterion = (
            torch.nn.CrossEntropyLoss()
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.005
        )
        self.store = MetricStore()

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x)
        loss = self.criterion(
            out[data.train_mask],
            data.y[data.train_mask],
        )
        loss_valid = self.criterion(
            out[data.val_mask],
            data.y[data.val_mask],
        )
        loss.backward()

        self.optimizer.step()
        self.model.eval()
        out = self.model(data.x)
        self.store.calculate_all(
            out.cpu(), data.y.cpu()
        )
        self.store.calculate_loss(out, data.y)
        return loss, loss_valid

    def train_full(self, data):
        stopper = EarlyStopperNaive(
            minimal_reduction=0.0001,
            patience_epochs=0,
        )
        for epoch in range(1, 101):
            train_loss, valid_loss = self.train(
                data
            )
            should_break, patience = stopper(
                valid_loss
            )
            print(
                f"Epoch: {epoch:03d}, Loss train: {train_loss:.4f}, Loss valid: {valid_loss}"
            )
            if should_break:
                break


# plot_plots()
# out_metrics_for_dataset['ANN'] = {
#     'accuracy': accuracies['test'][-1],
#     'f1_score': f1_scores['test'][-1]
# }
# print(out_metrics_for_dataset['ANN'])
# del model
# del optimizer
# del criterion
# del data
# free_gpu_cache()

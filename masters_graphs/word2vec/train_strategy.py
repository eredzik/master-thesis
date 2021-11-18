import datetime
import logging
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, roc_auc_score
from torch.nn.functional import embedding
from torch.utils.data import DataLoader

from model import NeuralNonlinearClassifier, NGramLanguageModeler


@dataclass
class KarateTrainNode2Vec:
    walk_length: int
    embedding_dim: int
    device: str
    n_nodes: int

    def __post_init__(self):
        self.loss_function = nn.NLLLoss()
        self.model = NGramLanguageModeler(
            self.n_nodes, self.embedding_dim, self.walk_length
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def forward_backward(self, context, target):
        context_idxs = torch.stack(context).to(self.device)
        self.model.zero_grad()
        log_probs = self.model.forward(context_idxs)
        self.loss = self.loss_function.forward(log_probs, target.to(self.device))
        self.loss.backward()
        self.optimizer.step()


@dataclass
class KarateTrainNeuralNonlinearClassifier:
    input_dim: int
    device: str
    output_n_classes: int = 2

    def __post_init__(self):

        self.loss_function = nn.NLLLoss()
        self.model = NeuralNonlinearClassifier(
            self.input_dim, n_classes=self.output_n_classes
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def forward_backward(self, input, target):

        self.model.zero_grad()
        probs = self.model.forward(input)
        self.loss = self.loss_function.forward(probs, target)
        self.loss.backward()
        self.optimizer.step()


def node2vec_train_strategy(walk_length, embedding_dim, dataset, n_classes_y):
    dataset_loader = DataLoader(dataset, batch_size=2048)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = datetime.datetime.now()
    logging.info(f"Starting train on Karate Club dataset at {start_time}")
    node2vec = KarateTrainNode2Vec(walk_length, embedding_dim, device, len(dataset))
    deepnn = KarateTrainNeuralNonlinearClassifier(
        embedding_dim, device, output_n_classes=n_classes_y
    )
    losses_node2vec = []
    losses_nn = []
    y_tensor = torch.tensor(dataset.y, dtype=torch.long, device=device)
    for epoch in range(100):
        total_loss = 0
        for index, sample in enumerate(dataset_loader):
            context, target = sample
            node2vec.forward_backward(context, target)
            total_loss += node2vec.loss.item()
            # print(index)
        losses_node2vec.append(total_loss)
        deepnn.forward_backward(node2vec.model.get_embeddings(), y_tensor)
        losses_nn.append(deepnn.loss.item())
        logging.info(f"Finished epoch {epoch}.")

    logging.info(
        f"Training finished on {datetime.datetime.now()}({datetime.datetime.now()-start_time})"
    )

    plt.subplot(3, 1, 1)
    plt.plot(losses_node2vec)
    plt.subplot(3, 1, 2)
    plt.plot(losses_nn)
    predictions = (
        deepnn.model.forward(node2vec.model.get_embeddings()).cpu().detach().numpy()
    )
    predictions_binary = predictions[:, 1] > predictions[:, 0]
    ax3 = plt.subplot(3, 1, 3)
    RocCurveDisplay.from_predictions(dataset.y, predictions_binary, ax=ax3)
    plt.show(block=True)

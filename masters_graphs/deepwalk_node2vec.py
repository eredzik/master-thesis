from typing import List
import networkx as nx
import random
from gensim.models import Word2Vec
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import numpy as np

from load_data import load_data, preprocess_features, run_regression


def load_adjlist(path: str):
    graph = nx.Graph()
    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line2 = line.split()
        source = int(line2[0])
        for target in line2[1:]:
            graph.add_edge(source, int(target))
    print(graph.nodes())
    print(graph.edges())
    return graph


# Source: https://towardsdatascience.com/deepwalk-its-behavior-and-how-to-implement-it-b5aac0290a15
def get_random_walk(graph: nx.Graph, node: int, n_steps: int) -> List[str]:
    """Given a graph and a node,
    return a random walk starting from the node
    """
    local_path = [
        str(node),
    ]
    target_node = node

    for _ in range(n_steps):
        neighbors = list(nx.all_neighbors(graph, target_node))
        if neighbors:
            target_node = random.choice(neighbors)
            local_path.append(str(target_node))
    return local_path


def get_walk_paths(
    graph: nx.Graph,
    walks_per_node: int = 10,
    n_steps_per_walk: int = 5,
):
    walks = []
    for node in graph.nodes():
        for _ in range(walks_per_node):
            walks.append(get_random_walk(graph, node, n_steps_per_walk))

    return walks


def build_model(walks, progress_per, epochs=20, min_count=0, workers=4):
    embedder = Word2Vec(
        window=10, sg=1, seed=521, vector_size=2, min_count=min_count, workers=workers
    )
    embedder.build_vocab(walks, progress_per=progress_per)
    embedder.train(walks, total_examples=embedder.corpus_count, epochs=epochs)
    return embedder


def run_zachary():
    gf = nx.karate_club_graph()
    walks = get_walk_paths(gf)
    embedder = build_model(walks)

    logreg = SGDClassifier(loss="log", n_jobs=10)
    labels = np.array([int(y["club"] == "Mr. Hi") for _, y in gf.nodes(data=True)])
    embeddings = [embedder.wv[str(node)] for node in gf.nodes.keys()]
    logreg.fit(embeddings, labels)
    print("Validation scores")
    print(f1_score(labels, logreg.predict(embeddings), average="micro"))


def run_reddit():
    dataset_dir = "data/reddit"
    train_ids, test_ids, val_ids, train_labels, test_labels, val_labels, gf = load_data(
        dataset_dir
    )

    train_feats, test_feats, val_feats = preprocess_features(
        dataset_dir, train_ids, test_ids, val_ids
    )
    walks = get_walk_paths(gf)
    print("Running regression..")
    run_regression(
        train_feats, train_labels, test_feats, test_labels, val_feats, val_labels
    )


def main():
    run_zachary()
    pass


if __name__ == "__main__":
    main()

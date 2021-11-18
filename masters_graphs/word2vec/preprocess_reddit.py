import json
import masters_graphs.networkx_nodelink_legacy as json_graph
from tap import Tap
import networkx as nx
import sys


class Args(Tap):
    graph_dataset: str
    classes_dataset: str
    train_output_dataset: str
    test_output_dataset: str
    validation_output_dataset: str


#  + "/reddit-G.json"
# "/reddit-class_map.json"
def preprocess_reddit(args: Args):
    with open(args.graph_dataset) as f:
        js = json.load(f)
        gf = json_graph.node_link_graph(js)
    with open(args.classes_dataset) as f:
        labels = json.load(f)
        train_nodes = gf.subgraph(
            [
                n
                for n in gf.nodes()
                if not gf.nodes(data=True)[n].get("val", False)
                and not gf.nodes(data=True)[n].get("test", False)
            ]
        ).copy()
        # TODO: How train test split should be done?
        # https://arxiv.org/pdf/1811.12159.pdf
        # 2.1 - talks about link prediction task, not node classification
        # https://datascience.stackexchange.com/questions/99706/how-train-test-split-works-for-graph-neural-networks
        # Above issue tackles exact same problem
        # Result: There are two problems at hand:
        # Transductive learning
        # (more like semisupervised - I know whole graph but don't know labels)
        # Inductive learning
        # Train and test are completely separated
        nx.write_gpickle(train_nodes, args.train_output_dataset)
        test_nodes = gf.subgraph(
            [n for n in gf.nodes() if gf.nodes(data=True)[n].get("test", False)]
        ).copy()
        nx.write_gpickle(test_nodes, args.test_output_dataset)

        validation_nodes = gf.subgraph(
            [n for n in gf.nodes() if gf.nodes(data=True)[n].get("val", False)]
        ).copy()
        nx.write_gpickle(validation_nodes, args.validation_output_dataset)


if __name__ == "__main__":
    preprocess_reddit(Args().parse_args())

import networkx as nx
import numpy as np
from torch.utils.data import Dataset
from networkx.readwrite import gml
from preprocessing import get_random_walk
import masters_graphs.networkx_nodelink_legacy as json_graph
import json
import enum


class DataSubset(str, enum.Enum):
    train = "train"
    test = "test"
    validation = "val"


class RedditDataset(Dataset):
    def __init__(self, dataset_dir, walk_length, dataset_mode: DataSubset):
        self.walk_length = walk_length
        with open(dataset_dir + "/reddit-G.json") as f:
            js = json.load(f)
            self.gf = json_graph.node_link_graph(js)
        with open(dataset_dir + "/reddit-class_map.json") as f:
            labels = json.load(f)
        if dataset_mode == DataSubset.train:
            self.nodes = [
                n
                for n in self.gf.nodes()
                if not self.gf.nodes(data=True)[n].get("val", False)
                and not self.gf.nodes(data=True)[n].get("test", False)
            ]
        elif dataset_mode == DataSubset.test:
            self.nodes = [
                n
                for n in self.gf.nodes()
                if self.gf.nodes(data=True)[n].get("test", False)
            ]
        elif dataset_mode == DataSubset.validation:
            self.nodes = [
                n
                for n in self.gf.nodes()
                if self.gf.nodes(data=True)[n].get("val", False)
            ]
        else:
            raise Exception(
                f"Wrong dataset_mode supplied ({dataset_mode}). Should be one of {DataSubset.__members__}"
            )
        self.y = [labels[i] for i in self.nodes]
        self.nodemap = {i: n for i, n in enumerate(self.nodes)}
        self.reversenodemap = {n: i for i, n in self.nodemap.items()}
        self.gf.remove_nodes_from([n for n in self.gf if n not in self.nodes])

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        walk, node = get_random_walk(self.gf, self.nodemap[idx], self.walk_length)
        encoded_walk = [self.encode_node(n) for n in walk]
        encoded_node = self.encode_node(node)
        return encoded_walk, encoded_node

    def encode_node(self, node):
        return self.reversenodemap[node]

    def write_gml(self, path):
        gml.write_gml(self.gf, path)

    @classmethod
    def from_gml(cls, path):
        inst = cls.__new__(cls)
        # super(RedditDataset, inst).__init__()
        inst.gf = gml.read_gml(path)
        return inst

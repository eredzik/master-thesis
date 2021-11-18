import networkx as nx
import numpy as np
from torch.utils.data import Dataset

from preprocessing import get_random_walk


class KarateClubDataset(Dataset):
    def __init__(self, walk_length):
        self.walk_length = walk_length
        self.gf = nx.karate_club_graph()
        self.vocab = list(self.gf.nodes())
        self.y = np.array([n[1]["club"] == "Mr. Hi" for n in self.gf.nodes(data=True)])

    def __len__(self):
        return len(self.gf.nodes())

    def __getitem__(self, idx):
        return get_random_walk(self.gf, idx, self.walk_length)

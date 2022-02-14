import random
import networkx as nx


def get_random_walks(gf: nx.Graph, walk_length, walk_number_per_node):
    walks = []
    for node in gf.nodes():
        for _ in range(walk_number_per_node):
            walk = []
            curr = node
            for _ in range(walk_length):
                curr = random.choice(list(gf.adj[curr].keys()))
                walk.append(curr)
            walks.append((walk, node))
    return walks


def get_random_walk(gf: nx.Graph, node_idx, walk_length):
    walk = []
    curr = node_idx
    for _ in range(walk_length):
        if len(gf.adj[curr]) == 0:
            walk.append(curr)
        else:
            curr = random.choice(list(gf.adj[curr].keys()))
            walk.append(curr)
    return walk, node_idx

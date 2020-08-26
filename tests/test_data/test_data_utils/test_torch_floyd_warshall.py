import torch

from caldera.data import GraphData
from caldera.data.utils import floyd_warshall


def test_floyd_warshall():
    data = GraphData.random(5, 4, 3, min_nodes=1000, min_edges=1000)
    W = floyd_warshall(data)
    print(W)


import numpy as np


# TODO: fast find neighbors
def test_find_neighbors():
    data = GraphData.random(5, 4, 3, min_nodes=1000, min_edges=1000)
    W = floyd_warshall(data)
    print(W.__class__)
    nodes = torch.LongTensor([[0, 1, 2, 3]])

    x = W[nodes]
    noninf = x != float("inf")
    reachable = x <= 3

    print(x.__class__)
    print(noninf.__class__)
    c = np.logical_and(noninf, reachable)
    d = np.where(c)
    print(d)
    neighbors = d[1]
    print(neighbors)

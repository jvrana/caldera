import torch

from caldera.data import GraphData
from caldera.data.utils import floyd_warshall


def test_floyd_warshall():
    data = GraphData.random(5, 4, 3, min_nodes=1000, min_edges=1000)
    W = floyd_warshall(data)
    print(W)


def test_find_neighbors():
    data = GraphData.random(5, 4, 3, min_nodes=1000, min_edges=1000)
    W = floyd_warshall(data)

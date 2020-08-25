from caldera.data.utils import torch_floyd_warshall
from caldera.data import GraphData, GraphBatch


def test_floyd_warshall():
    data = GraphData.random(5, 4, 3, min_nodes=1000, min_edges=1000)

    torch_floyd_warshall(data)

from caldera.data.utils import floyd_warshall
from caldera.data import GraphData, GraphBatch


def test_floyd_warshall():
    data = GraphData.random(5, 4, 3, min_nodes=1000, min_edges=1000)
    W = floyd_warshall(data)
    print(W)

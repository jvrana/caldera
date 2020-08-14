from caldera.data import GraphBatch, GraphData
from caldera.transforms import FullyConnected
from caldera.utils import deterministic_seed
from caldera.data.utils import _edges_to_tuples_set
import networkx as nx
import torch





def test_fully_connected_singe_graph_batch_manual():
    deterministic_seed(0)
    x = torch.randn((3, 1))
    e = torch.randn((2, 2))
    g = torch.randn((3, 1))
    edges = torch.tensor([
        [0, 1],
        [0, 1]
    ])
    data = GraphData(x, e, g, edges)
    batch = GraphBatch.from_data_list([data, data])
    batch2 = FullyConnected()(batch)
    print(batch2.edges)
    assert batch2.edges.shape[1] == 18
    edges_set = _edges_to_tuples_set(batch2.edges)
    assert len(edges_set) == 18


def test_fully_connected_singe_graph_batch():
    deterministic_seed(0)
    data = GraphData.random(5, 4, 3)
    batch = GraphBatch.from_data_list([data])
    t = FullyConnected()
    batch2 = t(batch)
    assert batch2.edges.shape[1] > batch.edges.shape[1]


def test_fully_connected_graph_batch():
    deterministic_seed(0)
    batch = GraphBatch.random_batch(10000, 5, 4, 3)
    t = FullyConnected()
    batch2 = t(batch)
    assert batch2.edges.shape[1] > batch.edges.shape[1]


def test_fully_connected_singe_graph_batch():
    deterministic_seed(0)
    data = GraphData.random(5, 4, 3)
    t = FullyConnected()
    data2 = t(data)
    assert data2.edges.shape[1] > data.edges.shape[1]

# test GraphData and GraphBatch result in differentiable
# ^^ is same
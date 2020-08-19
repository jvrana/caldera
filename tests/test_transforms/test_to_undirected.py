from caldera.data import GraphBatch, GraphData
from caldera.transforms import Undirected
from caldera.utils import deterministic_seed
from caldera.data.utils import _edges_to_tuples_set
import networkx as nx
import torch


def test_undirected_singe_graph_data_manual():

    deterministic_seed(0)

    x = torch.randn((4, 1))
    e = torch.randn((4, 2))
    g = torch.randn((3, 1))

    edges = torch.tensor([
        [0, 1, 2, 1],
        [1, 2, 3, 0]
    ])

    data = GraphData(x, e, g, edges)

    undirected = Undirected()

    undirected_data = undirected(data)
    assert undirected_data.edges.shape[1] == 6


def test_undirected_singe_graph_batch_manual():

    deterministic_seed(0)

    x = torch.randn((4, 1))
    e = torch.randn((4, 2))
    g = torch.randn((3, 1))

    edges = torch.tensor([
        [0, 1, 2, 1],
        [1, 2, 3, 0]
    ])



    data = GraphData(x, e, g, edges)

    batch = GraphBatch.from_data_list([data])

    undirected = Undirected()

    undirected_batch = undirected(batch)
    assert undirected_batch.edges.shape[1] == 6


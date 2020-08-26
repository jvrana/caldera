import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data.utils import to_coo_matrix


def test_to_coo_matrix():
    data = GraphData.random(5, 4, 3, min_edges=1000, min_nodes=1000)
    W = to_coo_matrix(data)
    assert W.size() == torch.Size([data.num_edges, data.num_edges, 4])


def test_to_coo_matrix_with_fill():
    data = GraphData.random(5, 4, 3, min_edges=1000, min_nodes=1000)
    W = to_coo_matrix(data, fill_value=1)
    assert W.size() == torch.Size([data.num_edges, data.num_edges])

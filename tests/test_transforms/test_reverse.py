import torch
from caldera.transforms import Reverse
from caldera.data import GraphBatch, GraphData


def test_reverse_graph_data():
    data = GraphData.random(5, 4, 3)
    reverse = Reverse()
    reversed_data = reverse(data)
    assert torch.all(reversed_data.edges.flip(1) == data.edges)


def test_reverse_graph_batch():
    data = GraphBatch.random_batch(100, 5, 4, 3)
    reverse = Reverse()
    reversed_data = reverse(data)
    assert torch.all(reversed_data.edges.flip(1) == data.edges)
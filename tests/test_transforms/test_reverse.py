import torch
from caldera.transforms import Reverse
from caldera.data import GraphBatch, GraphData


def test_reverse_graph_data(random_data):
    reverse = Reverse()
    reversed_data = reverse(random_data)
    assert torch.all(reversed_data.edges.flip(1) == random_data.edges)


def test_reverse_graph_batch(random_data):
    reverse = Reverse()
    reversed_data = reverse(random_data)
    assert torch.all(reversed_data.edges.flip(1) == random_data.edges)

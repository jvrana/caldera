import torch
from caldera.transforms import Shuffle
from caldera.data import GraphBatch


def test_reverse_graph_data(random_data):
    shuffle = Shuffle()
    shuffled = shuffle(random_data)
    assert not torch.allclose(shuffled.x, random_data.x)
    assert not torch.allclose(shuffled.e, random_data.e)
    assert not torch.allclose(shuffled.edges, random_data.edges)
    if random_data.__class__ is GraphBatch:
        assert not torch.allclose(shuffled.g, random_data.g)
        assert not torch.all(shuffled.node_idx == random_data.node_idx)
        assert not torch.all(shuffled.edge_idx == random_data.edge_idx)
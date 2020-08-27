import pytest
import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.transforms import Undirected
from caldera.utils import deterministic_seed


@pytest.fixture(params=[GraphData, GraphBatch])
def data(request):
    data_cls = request.param
    deterministic_seed(0)

    x = torch.randn((4, 1))
    e = torch.randn((4, 2))
    g = torch.randn((3, 1))

    edges = torch.tensor([[0, 1, 2, 1], [1, 2, 3, 0]])

    data = GraphData(x, e, g, edges)
    if data_cls is GraphBatch:
        return GraphBatch.from_data_list([data])
    else:
        return data


def test_undirected(data):
    undirected = Undirected()
    undirected_data = undirected(data)
    assert undirected_data.edges.shape[1] == 6

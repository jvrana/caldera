from caldera.data import GraphBatch, GraphData
from caldera.transforms import AddSelfLoops
from caldera.utils import deterministic_seed
import torch
import pytest


@pytest.fixture(params=[
    GraphData, GraphBatch
])
def data(request):

    data_cls = request.param
    deterministic_seed(0)

    x = torch.randn((4, 1))
    e = torch.randn((5, 2))
    g = torch.randn((3, 1))

    edges = torch.tensor([
        [0, 1, 2, 1, 0],
        [1, 2, 3, 0, 0]
    ])

    data = GraphData(x, e, g, edges)
    if data_cls is GraphBatch:
        return GraphBatch.from_data_list([data])
    else:
        return data


def test_self_loop(data):
    add_self_loops = AddSelfLoops()
    data2 = add_self_loops(data)
    assert data2.edges.shape[1] == 8
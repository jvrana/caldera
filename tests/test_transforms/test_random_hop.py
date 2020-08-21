from caldera.transforms import RandomHop
from caldera.data import GraphBatch, GraphData
from caldera.data.utils import get_edge_dict
import pytest


@pytest.mark.parametrize(
    "random_data",
    [
        (GraphData, None, None, {"max_nodes": 0}),
        (GraphData, None, None, {"max_nodes": 0}),
        (GraphBatch, None, None, {"min_edges": 1, "min_nodes": 1}),
        (GraphBatch, None, None, {"min_edges": 1, "min_nodes": 1}),
    ],
    ids=lambda x: x[0].__name__,
    indirect=True,
)
@pytest.mark.parametrize(
    "seeds", list(range(10)), ids=lambda x: "seed" + str(x), indirect=True
)
def test_random_hop(random_data, seeds):
    hop = RandomHop(1, 2)
    hopped = hop(random_data)
    print(hopped.shape)
    print(hopped.size)


@pytest.mark.parametrize(
    "random_data",
    [(GraphBatch, None, (1000, 100, 50, 25), {"min_edges": 1, "min_nodes": 1})],
    ids=lambda x: x[0].__name__,
    indirect=True,
)
def test_benchmark_hop(random_data):
    hop = RandomHop(1, 2)
    edge_dict = get_edge_dict(random_data.edges)
    for i in range(100):
        hop(random_data, edge_dict=edge_dict)

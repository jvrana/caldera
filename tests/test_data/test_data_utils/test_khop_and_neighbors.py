import networkx as nx
import pytest
import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data.utils import induce
from caldera.data.utils import neighbors
from caldera.data.utils import tensor_induce
from caldera.utils import deterministic_seed
from caldera.utils.testing import nx_random_features


@pytest.fixture(params=[GraphData, GraphBatch])
def data(request):
    deterministic_seed(0)
    data_cls = request.param
    if data_cls is GraphData:
        return GraphData.random(5, 4, 3)
    else:
        return GraphBatch.random_batch(10, 5, 4, 3)


@pytest.mark.parametrize(
    "n", [0, 1, torch.tensor(0), torch.tensor(1), torch.LongTensor([0, 1, 2])]
)
def test_neighbors_runnable(data, n):
    neighbors(data, n)


@pytest.mark.parametrize(
    ("edges", "source", "kwargs", "expected"),
    [
        (
            torch.LongTensor([[0, 1, 2, 0], [1, 2, 3, 2]]),
            0,
            {},
            torch.LongTensor([1, 2]),
        ),
        (
            torch.LongTensor([[0, 1, 2, 0], [1, 2, 3, 2]]),
            torch.tensor(0),
            {},
            torch.LongTensor([1, 2]),
        ),
        (torch.LongTensor([[0, 1, 2, 0], [1, 2, 3, 2]]), 1, {}, torch.LongTensor([2])),
        (
            torch.LongTensor([[0, 1, 2, 0], [1, 2, 3, 2]]),
            1,
            {"undirected": True},
            torch.LongTensor([0, 2]),
        ),
        (
            torch.LongTensor([[0, 1, 2, 0], [1, 2, 3, 2]]),
            torch.tensor(1),
            {},
            torch.LongTensor([2]),
        ),
        (
            torch.LongTensor([[0, 1, 2, 0], [1, 2, 3, 2]]),
            torch.tensor([0, 1, 2]),
            {},
            torch.LongTensor([1, 2, 3]),
        ),
        (
            torch.LongTensor([[2, 0, 1, 0], [3, 1, 2, 2]]),
            torch.tensor([0, 1, 2]),
            {},
            torch.LongTensor([1, 2, 3]),
        ),
        (
            torch.LongTensor([[0, 1, 2, 0], [1, 2, 3, 2]]),
            torch.BoolTensor(
                [True, True, True, False, False, False, False, False, False, False]
            ),
            {},
            torch.BoolTensor(
                [False, True, True, True, False, False, False, False, False, False]
            ),
        ),
        (
            torch.LongTensor([[0, 1, 2, 0], [1, 2, 3, 2]]),
            torch.BoolTensor(
                [True, False, False, False, False, False, False, False, False, False]
            ),
            {},
            torch.BoolTensor(
                [False, True, True, False, False, False, False, False, False, False]
            ),
        ),
    ],
)
def test_neighbors(edges, source, kwargs, expected):
    data = GraphData.random(
        5,
        4,
        3,
        min_nodes=10,
        max_nodes=10,
        min_edges=edges.shape[1],
        max_edges=edges.shape[1],
    )
    data.edges = edges
    data.debug()

    res = neighbors(data, source, **kwargs)
    print(res)
    assert torch.all(res == expected)


@pytest.mark.parametrize(
    ("edges", "k", "source", "expected"),
    [
        (
            torch.LongTensor(
                [[0, 1], [0, 2], [2, 3], [1, 3], [3, 4], [3, 5], [5, 6]]
            ).T,
            1,
            torch.LongTensor([0]),
            torch.LongTensor([0, 1, 2]),
        ),
        (
            torch.LongTensor(
                [[0, 1], [0, 2], [2, 3], [1, 3], [3, 4], [3, 5], [5, 6]]
            ).T,
            2,
            torch.LongTensor([0]),
            torch.LongTensor([0, 1, 2, 3]),
        ),
        (
            torch.LongTensor(
                [[0, 1], [0, 2], [2, 3], [1, 3], [3, 4], [3, 5], [5, 6]]
            ).T,
            3,
            torch.LongTensor([0]),
            torch.LongTensor([0, 1, 2, 3, 4, 5]),
        ),
        (
            torch.LongTensor(
                [[0, 1], [0, 2], [2, 3], [1, 3], [3, 4], [3, 5], [5, 6]]
            ).T,
            3,
            torch.BoolTensor([True] + [False] * 9),
            torch.BoolTensor([True] * 6 + [False] * 4),
        ),
    ],
)
def test_k_hop(edges, k, source, expected):
    deterministic_seed(0)
    data = GraphData.random(
        5,
        4,
        3,
        min_nodes=10,
        max_nodes=10,
        min_edges=edges.shape[1],
        max_edges=edges.shape[1],
    )
    data.edges = edges
    data.debug()

    res = induce(data, source, k)
    print(res)
    assert torch.all(res == expected)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_k_hop_random_graph(k):
    g1 = nx.grid_graph(dim=[2, 3, 4])
    g2 = nx.grid_graph(dim=[2, 3, 4])
    nx_random_features(g1, 5, 4, 3)
    nx_random_features(g2, 5, 4, 3)
    batch = GraphBatch.from_networkx_list([g1, g2])

    nodes = torch.BoolTensor([False] * batch.num_nodes)
    nodes[0] = True
    node_mask = induce(batch, nodes, k)
    subgraph = batch.apply_node_mask(node_mask)
    print(subgraph.info())


def test_k_hop_random_graph_benchmark():
    k = 2
    batch = GraphBatch.random_batch(1000, 50, 20, 30)

    for _ in range(100):
        nodes = torch.full((batch.num_nodes,), False, dtype=torch.bool)
        idx = torch.randint(batch.num_nodes, (2,))
        nodes[idx] = True
        node_mask = tensor_induce(batch, nodes, k)
        subgraph = batch.apply_node_mask(node_mask)
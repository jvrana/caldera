from caldera.data import GraphData, GraphBatch
import torch
from caldera.utils import deterministic_seed
import networkx as nx
import pytest
from caldera.data.utils import nx_random_features


def test_mask_all_nodes():
    deterministic_seed(0)

    data = GraphData(
        torch.randn(5, 5),
        torch.randn(3, 2),
        torch.randn(1, 1),
        edges=torch.LongTensor([[0, 0, 0], [1, 2, 3]]),
    )
    node_mask = torch.BoolTensor([False, False, False, False, False])
    data2 = data.apply_node_mask(node_mask)
    assert data2.num_nodes == 0
    assert data2.num_edges == 0


def test_mask_no_nodes():
    deterministic_seed(0)

    data = GraphData(
        torch.randn(5, 5),
        torch.randn(3, 2),
        torch.randn(1, 1),
        edges=torch.LongTensor([[0, 0, 0], [1, 2, 3]]),
    )
    node_mask = torch.BoolTensor([True, True, True, True, True])
    data2 = data.apply_node_mask(node_mask)
    assert data2.num_nodes == 5
    assert data2.num_edges == 3


def test_mask_all_edges():
    deterministic_seed(0)

    data = GraphData(
        torch.randn(5, 5),
        torch.randn(3, 2),
        torch.randn(1, 1),
        edges=torch.LongTensor([[0, 0, 0], [1, 2, 3]]),
    )
    edge_mask = torch.BoolTensor([False, False, False])
    data2 = data.apply_edge_mask(edge_mask)
    assert data2.num_nodes == 5
    assert data2.num_edges == 0


def test_mask_no_edges():
    deterministic_seed(0)

    data = GraphData(
        torch.randn(5, 5),
        torch.randn(3, 2),
        torch.randn(1, 1),
        edges=torch.LongTensor([[0, 0, 0], [1, 2, 3]]),
    )
    edge_mask = torch.BoolTensor([True, True, True])
    data2 = data.apply_edge_mask(edge_mask)
    assert data2.num_nodes == 5
    assert data2.num_edges == 3


def test_mask_one_edges():
    deterministic_seed(0)

    edges = torch.LongTensor([[0, 0, 0], [1, 2, 3]])
    expected_edges = torch.LongTensor([[0, 0], [1, 3]])

    e = torch.randn(3, 2)
    edge_mask = torch.BoolTensor([True, False, True])
    eidx = torch.where(edge_mask)
    expected_e = e[eidx]

    data = GraphData(torch.randn(5, 5), e, torch.randn(1, 1), edges=edges)

    data2 = data.apply_edge_mask(edge_mask)
    assert torch.all(data2.edges == expected_edges)
    assert torch.all(data2.e == expected_e)
    assert torch.all(data2.g == data.g)
    assert torch.all(data2.x == data.x)


def test_mask_one_node():
    deterministic_seed(0)

    edges = torch.LongTensor([[0, 1, 0], [1, 2, 3]])
    expected_edges = torch.LongTensor([[0], [1]])

    node_mask = torch.BoolTensor([False, True, True, True, True])

    x = torch.randn(5, 5)
    expected_x = x[node_mask]

    e = torch.randn(3, 2)
    expected_e = e[torch.LongTensor([1])]

    data = GraphData(x, e, torch.randn(1, 1), edges=edges)

    data2 = data.apply_node_mask(node_mask)
    assert torch.all(data2.edges == expected_edges)

    print(data2.x)
    print(expected_x)
    assert torch.allclose(data2.x, expected_x)
    assert torch.allclose(data2.e, expected_e)
    assert torch.allclose(data2.g, data.g)


@pytest.fixture(params=[GraphData, GraphBatch])
def grid_data(request):
    def newg(g):
        return nx_random_features(g, 5, 4, 3)

    if request.param is GraphData:
        g = newg(nx.grid_graph([2, 4, 3]))
        return GraphData.from_networkx(g)
    elif request.param is GraphBatch:
        graphs = [newg(nx.grid_graph([2, 4, 3]))  for _ in range(10)]
        return GraphBatch.from_networkx_list(graphs)
    else:
        raise ValueError()


def test_node_mask_grid_graph(grid_data):
    print(grid_data.size)
    node_mask = torch.randint(2, (grid_data.num_nodes,), dtype=torch.bool)
    subgraph = grid_data.apply_node_mask(node_mask)
    assert subgraph.__class__ is grid_data.__class__
    print(subgraph.size)
from caldera.data import GraphData
import torch
from caldera.utils import deterministic_seed


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

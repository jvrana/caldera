from caldera.data import GraphData, GraphBatch
import torch
import pytest


@pytest.fixture(params=[GraphData, GraphBatch])
def data(request):
    args = (5, 4, 3)
    kwargs = dict(min_nodes=5, max_nodes=5, min_edges=5, max_edges=5)
    if request.param is GraphData:
        return GraphData.random(*args, **kwargs)
    else:
        return GraphBatch.random_batch(100, *args, **kwargs)


@pytest.fixture
def shuffle(request):
    method, inplace = request.param

    if inplace:

        def wrapped(data):
            data1 = data.clone()
            getattr(data, method + "_")()
            return data1, data

    else:

        def wrapped(data):
            data2 = getattr(data, method)()
            return data, data2

    return wrapped


@pytest.mark.parametrize(
    "shuffle", [("shuffle_nodes", True), ("shuffle_nodes", False),], indirect=True
)
def test_shuffle_nodes(data, shuffle):
    data1, data2 = shuffle(data)

    assert torch.all(data1.x[data1.edges.T] == data2.x[data2.edges.T])
    assert torch.all(data1.e == data2.e)
    assert torch.all(data1.g == data2.g)
    assert not torch.all(data1.x == data2.x)
    assert not torch.all(data1.edges == data2.edges)


@pytest.mark.parametrize(
    "shuffle", [("shuffle_edges", True), ("shuffle_edges", False),], indirect=True
)
def test_shuffle_edges(data, shuffle):
    data1, data2 = shuffle(data)

    assert not torch.all(data1.e == data2.e)
    assert torch.all(data1.g == data2.g)
    assert torch.all(data1.x == data2.x)
    assert not torch.all(data1.edges == data2.edges)


@pytest.mark.parametrize(
    "shuffle", [("shuffle_graphs", True), ("shuffle_graphs", False),], indirect=True
)
def test_shuffle_graphs(shuffle):
    args = (5, 4, 3)
    kwargs = dict(min_nodes=5, max_nodes=5, min_edges=5, max_edges=5)
    data = GraphBatch.random_batch(100, *args, **kwargs)
    data1, data2 = shuffle(data)
    if data.__class__ is GraphData:
        pytest.xfail("GraphData has no `shuffle_graphs` method")

    assert torch.all(data1.e == data2.e)
    assert not torch.all(data1.g == data2.g)
    assert torch.all(data1.x == data2.x)
    assert torch.all(data1.edges == data2.edges)
    assert not torch.all(data1.node_idx == data2.node_idx)
    assert not torch.all(data1.edge_idx == data2.edge_idx)


@pytest.mark.parametrize(
    "shuffle", [("shuffle", True), ("shuffle", False),], indirect=True
)
def test_shuffle(data, shuffle):
    data1, data2 = shuffle(data)
    assert not torch.all(data1.e == data2.e)
    assert not torch.all(data1.x == data2.x)
    assert not torch.all(data1.edges == data2.edges)
    if data.__class__ is GraphBatch:
        assert not torch.all(data1.g == data2.g)
        assert not torch.all(data1.node_idx == data2.node_idx)
        assert not torch.all(data1.edge_idx == data2.edge_idx)

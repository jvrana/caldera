import pytest
import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data import GraphDataLoader


def test_loader():
    datalist = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist, batch_size=32, shuffle=True)

    for batch in loader:
        assert batch.size[2] == 32


def test_loader_zipped():
    datalist1 = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    datalist2 = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist1, datalist2, batch_size=32, shuffle=True)

    for a, b in loader:
        assert isinstance(a, GraphBatch)
        assert isinstance(b, GraphBatch)
        assert a is not b


def test_loader_mem_sizes():
    datalist = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist, batch_size=1, shuffle=True)
    print(loader.mem_sizes())
    print(loader.mem_sizes().to(torch.float).std())


def test_loader_limit_mem_sizes():
    datalist = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist, batch_size=1, shuffle=True)

    assert not list(loader(limit_mem_size=10))
    assert list(loader(limit_mem_size=1000))


def test_loader_first():
    datalist = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist, batch_size=32, shuffle=True)

    batch = loader.first()
    assert isinstance(batch, GraphBatch)
    assert batch.shape == (5, 4, 3)
    assert batch.num_graphs == 32


def test_loader_multiple_cpus():
    datalist = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist, batch_size=32, shuffle=True, num_workers=2)
    for x in loader:
        print(x)


@pytest.mark.parametrize("shuffle", [True, False])
def test_loader_shuffle(shuffle):
    datalist = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    non_shuffled = GraphDataLoader(datalist, batch_size=32, shuffle=shuffle)
    for x in non_shuffled:
        pass
    for y in non_shuffled:
        pass
    if not shuffle:
        assert torch.allclose(x.x, y.x)
    else:
        assert x.x.size != y.x.size or not torch.allclose(x.x, y.x)

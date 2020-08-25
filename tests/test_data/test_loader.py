import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data import GraphDataLoader


def test_loader():
    datalist = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist, batch_size=32, shuffle=True)

    for batch in loader:
        assert batch.size[2] == 32


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

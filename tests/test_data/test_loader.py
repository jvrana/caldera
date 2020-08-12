from pyrographnets.data import GraphDataLoader, GraphData, GraphBatch


def test_loader():
    datalist = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist, batch_size=32, shuffle=True)

    for batch in loader:
        assert batch.size[2] == 32


def test_loader_first():
    datalist = [GraphData.random(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist, batch_size=32, shuffle=True)

    batch = loader.first()
    assert isinstance(batch, GraphBatch)
    assert batch.shape == (5, 4, 3)
    assert batch.num_graphs == 32

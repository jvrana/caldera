from caldera.data import GraphDataset, GraphBatchDataset
from caldera.data import GraphData, GraphBatch
from caldera.data import GraphDataLoader


def test_loader_dataset():
    datalist = [GraphData.random(5, 4, 3) for _ in range(32*4)]
    dataset = GraphDataset(datalist)

    for batch in GraphDataLoader(dataset, shuffle=True, batch_size=32):
        print(batch.size)
        assert isinstance(batch, GraphBatch)
        assert batch.size[-1] == 32



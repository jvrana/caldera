from pyrographnets.data import GraphDataLoader
from pyrographnets.data.utils import random_data


def test_loader():
    datalist = [random_data(5, 4, 3) for _ in range(32 * 5)]
    loader = GraphDataLoader(datalist, batch_size=32, shuffle=True)

    for batch in loader:
        assert batch.size[2] == 32
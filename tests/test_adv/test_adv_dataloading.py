import pytest

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.dataset import GraphDataset
from caldera.transforms import RandomHop


# test on medium sized graphs
@pytest.mark.parametrize(
    "random_data_list",
    [(10, GraphData, None, (5, 4, 3), {"min_nodes": 1000})],
    indirect=True,
)
def test_random_k_hop(random_data_list):
    transform = RandomHop(10, 2)

    # TODO: better way to do transforms on large data

    new_data = []
    for _ in range(100):
        for data in GraphDataset(random_data_list, transform=transform):
            new_data.append(data)

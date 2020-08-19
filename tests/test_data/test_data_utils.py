import pytest
from caldera.data import GraphBatch, GraphData
from caldera.data.utils import neighbors
from caldera.data.utils import hop
from caldera.utils import deterministic_seed

import torch


@pytest.fixture(params=[GraphData, GraphBatch])
def data(request):
    deterministic_seed(0)
    data_cls = request.param
    if data_cls is GraphData:
        return GraphData.random(5, 4, 3)
    else:
        return GraphBatch.random_batch(10, 5, 4, 3)

@pytest.mark.parametrize('n', [
    0,
    1,
    torch.tensor(0),
    torch.tensor(1),
    torch.LongTensor([0, 1, 2])
])
def test_neighbors(data, n):
    neighbors(data, n)



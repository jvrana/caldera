import pytest

from caldera.data import GraphBatch
from caldera.transforms import AddSelfLoops
from caldera.transforms import FullyConnected
from caldera.transforms import RandomEdgeMask
from caldera.transforms import RandomHop
from caldera.transforms import RandomNodeMask
from caldera.transforms import Reverse
from caldera.transforms import Shuffle
from caldera.transforms import Undirected


transforms = [
    AddSelfLoops(),
    FullyConnected(),
    RandomNodeMask(0.2),
    RandomEdgeMask(0.2),
    RandomHop(1, 2),
    Reverse(),
    Undirected(),
    Shuffle(),
]


@pytest.fixture(params=[1000])
def data(seeds, request):
    data = GraphBatch.random_batch(request.param, 5, 4, 3)
    return data


@pytest.mark.parametrize("transform", transforms, ids=lambda x: str(x))
def test_transform_benchmark(transform, data):
    assert transform(data)


@pytest.mark.parametrize("transform", transforms, ids=lambda x: str(x))
@pytest.mark.parametrize("data", [100], indirect=True)
def test_transform_does_not_share_storage(transform, data):
    data2 = transform(data)
    assert not data2.share_storage(data)
    assert not data.share_storage(data2)

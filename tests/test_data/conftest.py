import pytest

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils import deterministic_seed


# TODO: remove? in main conftest.py
@pytest.fixture(
    params=[
        (GraphData, (5, 4, 3)),
        (GraphData, (5, 4, 3), dict(min_edges=0, max_edges=0)),
        (GraphData, (5, 4, 3), dict(min_nodes=0, max_nodes=0)),
        (GraphData, (5, 7, 10)),
        (GraphBatch, (10, 5, 6, 7)),
        (GraphBatch, (10, 5, 6, 7), dict(min_edges=0, max_edges=0)),
        (GraphBatch, (10, 5, 6, 7), dict(min_nodes=0, max_nodes=0)),
        (GraphBatch, (100, 5, 6, 7)),
    ]
)
def random_data_example(request):
    deterministic_seed(0)
    if len(request.param) == 3:
        cls, args, kwargs = request.param
    elif len(request.param) == 2:
        cls, args, kwargs = request.param[0], request.param[1], {}
    if cls is GraphData:
        graph_data = cls.random(*args, **kwargs)
        return graph_data
    elif cls is GraphBatch:
        batch = cls.random_batch(*args, **kwargs)
        return batch
    else:
        raise Exception("Parameter not acceptable: {}".format(request.param))

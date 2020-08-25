import pytest

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.transforms import RandomEdgeMask
from caldera.transforms import RandomNodeMask


parametrize_dropout = pytest.mark.parametrize(
    "dropout",
    [
        pytest.param(
            -0.1,
            marks=pytest.mark.xfail(
                raises=ValueError, strict=True, reason="Dropout must be between 0 and 1"
            ),
        ),
        pytest.param(
            1.1,
            marks=pytest.mark.xfail(
                raises=ValueError, strict=True, reason="Dropout must be between 0 and 1"
            ),
        ),
        0.5,
        0.0,
        1.0,
    ],
    ids=lambda x: "dropout=" + str(x),
)

parametrize_random_data = pytest.mark.parametrize(
    "random_data",
    [(GraphData, None), (GraphBatch, None)],
    indirect=True,
    ids=lambda x: x[0].__name__,
)


@parametrize_dropout
@parametrize_random_data
def test_random_node_mask(random_data, dropout, seeds):
    transform = RandomNodeMask(dropout)
    data2 = transform(random_data)
    if dropout == 1:
        assert data2.num_nodes == 0
        assert data2.num_edges == 0
    elif dropout == 0:
        assert data2.num_nodes == random_data.num_nodes
        assert data2.num_edges == random_data.num_edges
    else:
        assert data2.num_nodes <= random_data.num_nodes
        assert data2.num_edges <= random_data.num_edges


@parametrize_dropout
@parametrize_random_data
def test_random_edge_mask(random_data, dropout, seeds):
    transform = RandomEdgeMask(dropout)
    data2 = transform(random_data)
    assert data2.num_nodes == random_data.num_nodes
    if dropout == 1:
        assert data2.num_edges == 0
    elif dropout == 0:
        assert data2.num_edges == random_data.num_edges
    else:
        assert data2.num_edges <= random_data.num_edges

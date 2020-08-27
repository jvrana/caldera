import pytest

from caldera.data.utils import graph_matrix


@pytest.mark.parametrize("include_edge_attr", [True, False])
def test_adj_matrix(random_data_example, include_edge_attr):
    M = graph_matrix(random_data_example, include_edge_attr=include_edge_attr)
    if include_edge_attr:
        assert M.ndimension() == 3
    else:
        assert M.ndimension() == 2

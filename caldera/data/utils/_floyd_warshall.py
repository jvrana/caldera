from typing import Union

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._shortest_path import (
    floyd_warshall as cs_graph_floyd_warshall,
)

from ._sparse import graph_data_to_coo_matrix
from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils.sparse import torch_coo_to_scipy_coo


def floyd_warshall(data: Union[GraphData, GraphBatch], **kwargs):
    """Run the floyd-warshall algorithm."""

    m = graph_data_to_coo_matrix(data, fill_value=1).coalesce()
    m._values()[:] = 1
    A = torch_coo_to_scipy_coo(m)
    graph = csr_matrix(A)

    default_kwargs = dict(directed=True)
    default_kwargs.update(kwargs)
    return cs_graph_floyd_warshall(graph, **default_kwargs)

from typing import Union

import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._shortest_path import (
    floyd_warshall as cs_graph_floyd_warshall,
)

from ._sparse import to_sparse_coo_matrix
from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils.sparse import torch_coo_to_scipy_coo


def floyd_warshall(data: Union[GraphData, GraphBatch], **kwargs) -> torch.Tensor:
    """Run the floyd-warshall algorithm."""
    m = to_sparse_coo_matrix(data, fill_value=1).coalesce()
    graph = csr_matrix(torch_coo_to_scipy_coo(m))
    default_kwargs = dict(directed=True, unweighted=True)
    default_kwargs.update(kwargs)
    matrix = cs_graph_floyd_warshall(graph, **default_kwargs)
    return torch.from_numpy(matrix)

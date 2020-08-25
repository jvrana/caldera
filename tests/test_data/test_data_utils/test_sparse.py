from typing import *

import torch

from caldera.data import *


# TODO: to COO matrix using roll index
def to_edge_attr_coo_matrix(data: Union[GraphData, GraphBatch]):
    edges = data.edges
    edge_attr = data.e
    ij = torch.cat(
        [
            edges.repeat(1, edge_attr.shape[-1]),
            torch.arange(edges.shape[1]).repeat(1, edge_attr.shape[-1]),
        ],
        dim=0,
    )
    if edge_attr.ndim == 1:
        v = edge_attr.repeat(edges.shape[1])
    else:
        v = edge_attr.view(-1)

    size = (data.num_nodes, data.num_nodes, edge_attr.shape[1])
    return torch.sparse_coo_tensor(ij, v, size=size, dtype=edge_attr.dtype)

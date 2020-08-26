from typing import Optional
from typing import Union

import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils.sparse import scatter_coo


def to_full_coo_matrix(
    data: Union[GraphData, GraphBatch], fill_value=1, dtype=torch.float,
):
    ij = data.edges
    v = torch.full(data.edges[0].shape, fill_value=fill_value, dtype=torch.float)
    size = torch.Size([data.num_nodes] * 2)
    return torch.sparse_coo_tensor(ij, v, size=size, dtype=dtype)


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


# TODO:
def to_coo_matrix():
    pass


def graph_data_to_coo_matrix(
    data: Union[GraphData, GraphBatch], fill_value=1, dtype=torch.float,
):
    ij = data.edges
    v = torch.full(data.edges[0].shape, fill_value=fill_value, dtype=torch.float)
    size = torch.Size([data.num_nodes] * 2)
    return torch.sparse_coo_tensor(ij, v, size=size, dtype=dtype)

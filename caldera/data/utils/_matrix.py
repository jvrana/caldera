from typing import Union

import torch

from caldera.data import GraphBatch
from caldera.data import GraphData


def graph_matrix(
    g: GraphData,
    dtype=torch.float,
    include_edge_attr: bool = True,
    fill_value: Union[int, float, torch.Tensor] = 0,
    edge_value: Union[int, float, torch.Tensor] = 1,
):
    edges = g.edges
    if include_edge_attr:
        shape = (g.num_nodes, g.num_nodes, g.e.shape[1])
    else:
        shape = (g.num_nodes, g.num_nodes)
    M = torch.full(shape, fill_value=fill_value, dtype=dtype)
    if include_edge_attr:
        v = g.e
    else:
        v = edge_value
    M[edges.unbind()] = v
    return M


def _degree_matrix_from_edges(
    edges: torch.LongTensor, n_nodes: int, i: int
) -> torch.LongTensor:
    D = torch.zeros(n_nodes, dtype=torch.long)
    a, b = torch.unique(edges[i], return_counts=True)
    D[(a, a)] = b
    return D


def in_degree_matrix_from_edges(
    edges: torch.LongTensor, n_nodes: int
) -> torch.LongTensor:
    return _degree_matrix_from_edges(edges, n_nodes, 1)


def out_degree_matrix_from_edges(
    edges: torch.LongTensor, n_nodes: int
) -> torch.LongTensor:
    return _degree_matrix_from_edges(edges, n_nodes, 0)


def adj_matrix(data: Union[GraphData, GraphBatch]) -> torch.LongTensor:
    return adj_matrix_from_edges(data.edges, data.num_nodes)


def in_degree(data: Union[GraphData, GraphBatch]) -> torch.LongTensor:
    return in_degree_matrix_from_edges(data.edges, data.num_nodes)


def out_degree(data: Union[GraphData, GraphBatch]) -> torch.LongTensor:
    return out_degree_matrix_from_edges(data.edges, data.num_nodes)


def adj_matrix_from_edges(edges: torch.LongTensor, n_nodes: int) -> torch.LongTensor:
    A = torch.zeros((n_nodes, n_nodes), dtype=torch.long)
    for i in edges.T:
        A[i[0], i[1]] += 1
    return A

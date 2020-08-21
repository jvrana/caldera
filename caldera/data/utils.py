import networkx as nx
import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from typing import Union, List, Tuple, Callable, Set
from typing import overload
from caldera.utils import long_isin


def _create_all_edges(start: int, n_nodes: int) -> torch.LongTensor:
    """
    Create all edges starting from node index 'start' for 'n_nodes' additional nodes.

    E.g. _all_edges(5, 4) would create all edges of nodes 5, 6, 7, 8 with shape (2, 4)
    """
    a = torch.arange(start, start + n_nodes, dtype=torch.long)
    b = a.repeat_interleave(torch.ones(n_nodes, dtype=torch.long) * n_nodes)
    c = torch.arange(start, start + n_nodes, dtype=torch.long).repeat(n_nodes)
    return torch.stack([b, c], axis=0)


def _edges_to_tuples_set(edges: torch.LongTensor) -> Set[Tuple[int, int]]:
    edge_tuples = set()
    for e in edges.T:
        edge_tuples.add((e[0].item(), e[1].item()))
    return edge_tuples


def _validate_edges(edges: torch.LongTensor):
    assert edges.ndim == 2
    assert edges.shape[0] == 2


def _tuples_set_to_tensor(tuples: List[Tuple[int, int]]):
    return torch.tensor(list(zip(*list(tuples))), dtype=torch.long)


def _apply_to_edge_sets(
    edges1: torch.LongTensor,
    edges2: torch.LongTensor,
    func: Callable[[Set[Tuple[int, int]], Set[Tuple[int, int]]], torch.LongTensor],
) -> torch.LongTensor:
    s1 = _edges_to_tuples_set(edges1)
    s2 = _edges_to_tuples_set(edges2)
    s3 = func(s1, s2)
    return _tuples_set_to_tensor(s3)


def edges_difference(e1: torch.LongTensor, e2: torch.LongTensor) -> torch.LongTensor:
    def difference(e1, e2):
        return e1.difference(e2)

    return _apply_to_edge_sets(e1, e2, difference)


def edges_intersection(e1: torch.LongTensor, e2: torch.LongTensor) -> torch.LongTensor:
    def intersection(e1, e2):
        return e1.intersection(e2)

    return _apply_to_edge_sets(e1, e2, intersection)


def _edge_difference(edges1, edges2):
    s1 = _edges_to_tuples_set(edges1)
    s2 = _edges_to_tuples_set(edges2)
    s3 = s1.difference(s2)

    return s1.difference(s2)


@overload
def add_edges(data: GraphBatch, fill_value: ..., kind: ...) -> GraphBatch:
    """
    Adds edges to the :class:`caldera.data.GraphBatch` instance.
    """
    ...


def add_edges(data: GraphData, fill_value: Union[float, int], kind: str) -> GraphData:
    """
    Adds edges to the :class:`caldera.data.GraphData`.

    :param data: :class:`caldera.data.GraphData` instance.
    :param fill_value: fill value for edge attribute tensor.
    :param kind: Choose from "self" (for self edges), "complete" (for complete graph) or "undirected" (undirected edges)
    :return:
    """
    UNDIRECTED = "undirected"
    COMPLETE = "complete"
    SELF = "self"
    VALID_KIND = [UNDIRECTED, COMPLETE, SELF]
    if kind not in VALID_KIND:
        raise ValueError("'kind' must be one of {}".format(VALID_KIND))

    data_cls = data.__class__
    if issubclass(data_cls, GraphBatch):
        node_idx = data.node_idx
        edge_idx = data.edge_idx
    elif issubclass(data_cls, GraphData):
        node_idx = torch.zeros(data.x.shape[0], dtype=torch.long)
        edge_idx = torch.zeros(data.e.shape[0], dtype=torch.long)
    else:
        raise ValueError(
            "data must be a subclass of {} or {}".format(
                GraphBatch.__class__.__name__, GraphData.__class__.__name__
            )
        )

    with torch.no_grad():
        # we count the number of nodes in each graph using node_idx
        gidx, n_nodes = torch.unique(node_idx, return_counts=True, sorted=True)
        _, n_edges = torch.unique(edge_idx, return_counts=True, sorted=True)

        eidx = 0
        nidx = 0
        graph_edges_list = []
        new_edges_lengths = torch.zeros(gidx.shape[0], dtype=torch.long)

        for _gidx, _n_nodes, _n_edges in zip(gidx, n_nodes, n_edges):
            graph_edges = data.edges[:, eidx : eidx + _n_edges]
            if kind == UNDIRECTED:
                missing_edges = edges_difference(graph_edges.flip(0), graph_edges)
            elif kind == COMPLETE:
                all_graph_edges = _create_all_edges(nidx, _n_nodes)
                missing_edges = edges_difference(all_graph_edges, graph_edges)
            elif kind == SELF:
                self_edges = torch.cat(
                    [torch.arange(nidx, nidx + _n_nodes).expand(1, -1)] * 2
                )
                missing_edges = edges_difference(self_edges, graph_edges)
            graph_edges_list.append(missing_edges)

            if not missing_edges.shape[0]:
                new_edges_lengths[_gidx] = 0
            else:
                new_edges_lengths[_gidx] = missing_edges.shape[1]

            nidx += _n_nodes
            eidx += _n_edges

        new_edges = torch.cat(graph_edges_list, axis=1)
        new_edge_idx = gidx.repeat_interleave(new_edges_lengths)
        new_edge_attr = torch.full(
            (new_edges.shape[1], data.e.shape[1]), fill_value=fill_value
        )

        edges = torch.cat([data.edges, new_edges], axis=1)
        edge_idx = torch.cat([edge_idx, new_edge_idx])
        edge_attr = torch.cat([data.e, new_edge_attr], axis=0)

        idx = edge_idx.argsort()

    if issubclass(data_cls, GraphBatch):
        return GraphBatch(
            node_attr=data.x.detach().clone(),
            edge_attr=edge_attr[idx],
            global_attr=data.g.detach().clone(),
            edges=edges[:, idx],
            node_idx=data.node_idx.detach().clone(),
            edge_idx=edge_idx[idx],
        )
    elif issubclass(data_cls, GraphData):
        return data_cls(
            node_attr=data.x.detach().clone(),
            edge_attr=edge_attr[idx],
            global_attr=data.g.detach().clone(),
            edges=edges[:, idx],
        )


@overload
def neighbors(data: ..., nodes: torch.BoolTensor) -> torch.BoolTensor:
    ...


def neighbors(
    data: Union[GraphData, GraphBatch], nodes: torch.LongTensor
) -> torch.LongTensor:
    """
    Return the neighbors of the provided nodes.

    :param data:
    :param nodes:
    :return:
    """
    if isinstance(nodes, int):
        nodes = torch.LongTensor([nodes])
    elif nodes.dtype == torch.long and nodes.ndim == 0:
        nodes = nodes.expand(1)
    is_bool = False
    if nodes.dtype == torch.bool:
        is_bool = True
        nodes = torch.where(nodes)[0]
    reachable = long_isin(data.edges[0], nodes)
    dest = data.edges[1][reachable]
    if is_bool:
        dest = long_isin(torch.arange(data.num_nodes), dest)
    else:
        dest = torch.unique(dest, sorted=True)
    return dest


@overload
def hop(data: ..., nodes: torch.BoolTensor) -> torch.BoolTensor:
    ...


def hop(
    data: Union[GraphData, GraphBatch], nodes: torch.LongTensor, k: int
) -> torch.LongTensor:
    if isinstance(nodes, int):
        nodes = torch.LongTensor([nodes])
    elif nodes.dtype == torch.long and nodes.ndim == 0:
        nodes = nodes.expand(1)

    visited = nodes.clone()
    for _k in range(k):
        nodes = neighbors(data, nodes)
        if nodes.dtype == torch.bool:
            visited = torch.logical_or(visited, nodes)
            if visited.sum() >= data.num_nodes:
                break
        else:
            visited = torch.unique(torch.cat([visited, nodes]), sorted=True)
            if visited.shape[0] >= data.num_nodes:
                break
    return visited


def nx_random_features(g: nx.DiGraph, n_feat: int, e_feat: int, g_feat: int):
    for _, ndata in g.nodes(data=True):
        ndata["features"] = torch.randn(n_feat)
    for _, _, edata in g.edges(data=True):
        edata["features"] = torch.randn(e_feat)
    g.data = {"features": torch.randn(g_feat)}
    return g


def adj_matrix_from_edges(edges: torch.LongTensor, n_nodes: int) -> torch.LongTensor:
    A = torch.zeros(n_nodes, dtype=torch.long)
    for i in edges.T:
        A[i[0], i[1]] += 1
    return A


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

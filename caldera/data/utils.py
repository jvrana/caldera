from typing import Optional

import networkx as nx
import numpy as np
import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils import _first
from caldera.utils import scatter_group
from typing import Union, List, Tuple, Callable, Set

def to_graph_data(
    g: nx.DiGraph,
    n_node_feat: Optional[int] = None,
    n_edge_feat: Optional[int] = None,
    n_glob_feat: Optional[int] = None,
):
    if hasattr(g, "data"):
        gdata = g.data
    else:
        gdata = {}

    if n_node_feat is None:
        _, ndata = _first(g.nodes(data=True))
        n_node_feat = ndata["features"].shape[0]

    if n_edge_feat is None:
        _, _, edata = _first(g.edges(data=True))
        n_edge_feat = edata["features"].shape[0]

    if n_glob_feat is None:
        n_glob_feat = gdata["features"].shape[0]

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    node_attr = np.empty((n_nodes, n_node_feat))
    edge_attr = np.empty((n_edges, n_edge_feat))
    glob_attr = np.empty((1, n_glob_feat))

    nodes = sorted(list(g.nodes(data=True)))
    ndict = {}
    for i, (n, ndata) in enumerate(nodes):
        node_attr[i] = ndata["features"]
        ndict[n] = i

    edges = np.empty((2, n_edges))
    for i, (n1, n2, edata) in enumerate(g.edges(data=True)):
        edges[:, i] = np.array([ndict[n1], ndict[n2]])
        edge_attr[i] = edata["features"]

    glob_attr[0] = g.data["features"]

    return GraphData(
        torch.tensor(node_attr, dtype=torch.float),
        torch.tensor(edge_attr, dtype=torch.float),
        torch.tensor(glob_attr, dtype=torch.float),
        torch.tensor(edges, dtype=torch.long),
    )


def graph_batch_to_data_list(batch: GraphBatch):
    assert issubclass(type(batch), GraphBatch)
    gidx_n, node_attr = scatter_group(batch.x, batch.node_idx)
    gidx_e, edge_attr = scatter_group(batch.e, batch.edge_idx)
    gidx_edge, edges = scatter_group(batch.edges.T, batch.edge_idx)

    def to_dict(a, b):
        return dict(zip([x.item() for x in a], b))

    ndict = to_dict(gidx_n, node_attr)
    edict = to_dict(gidx_e, edge_attr)
    edgesdict = to_dict(gidx_edge, edges)
    datalist = []
    for k in ndict:
        _edges = edgesdict[k].T - edgesdict[k].min()

        data = GraphData(ndict[k], edict[k], batch.g[k], _edges)
        datalist.append(data)
    return datalist


def graph_data_to_nx(data: GraphData):
    g = nx.DiGraph()
    for n, ndata in enumerate(data.x):
        g.add_node(n, **{"features": ndata})
    for i, e in enumerate(data.edges.T):
        g.add_edge(e[0], e[1], **{"features": data.e[i]})
    g.data = {"features": data.g}
    return g


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
        func: Callable[[Set[Tuple[int, int]], Set[Tuple[int, int]]], torch.LongTensor]) -> torch.LongTensor:
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


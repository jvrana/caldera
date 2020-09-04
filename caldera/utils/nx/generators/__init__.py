"""generators.

Networkx graph generators
"""
import itertools
import random
import uuid
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import networkx as nx
import torch

from caldera.utils import functional as Fn
from caldera.utils._tools import _resolve_range
from caldera.utils.nx import nx_is_directed


def nx_random_features(g: nx.DiGraph, n_feat: int, e_feat: int, g_feat: int):
    for _, ndata in g.nodes(data=True):
        ndata["features"] = torch.randn(n_feat)
    for _, _, edata in g.edges(data=True):
        edata["features"] = torch.randn(e_feat)
    g.data = {"features": torch.randn(g_feat)}
    return g


def rand_n_nodes_n_edges(
    n_nodes: Union[int, Tuple[int, int]],
    n_edges: Optional[Union[int, Tuple[int, int]]] = None,
    density: Optional[Union[float, Tuple[float, float]]] = None,
):
    n = _resolve_range(n_nodes)
    if n_edges is None and density is None:
        raise ValueError("Either density or n_edges must be provided.")
    elif n_edges is None:
        d = _resolve_range(density)
        e = int((d * n * (n - 1)) / 2)
        e = max(1, e)
    else:
        e = _resolve_range(n_edges)
    return n, e


def random_graph(
    n_nodes: Union[int, Tuple[int, int]],
    n_edges: Optional[Union[int, Tuple[int, int]]] = None,
    density: Optional[Union[float, Tuple[float, float]]] = None,
    generator: Callable[[int, int], nx.Graph] = nx.generators.dense_gnm_random_graph,
    *args,
    **kwargs
):
    n, e = rand_n_nodes_n_edges(n_nodes, n_edges, density)
    return generator(n, e, *args, **kwargs)


def random_node(g, n=None):
    nodes = [random.choice(list(g.nodes)) for _ in range(n or 1)]
    if n is None:
        nodes = nodes[0]
    return nodes


def random_edge(g, n=None):
    edges = [random.choice(list(g.edges)) for _ in range(n or 1)]
    if n is None:
        edges = edges[0]
    return edges


def _possible_edges(n1: Set, n2: Set, directed: bool, self_loops: bool = False):
    """Compute the number of possible edges between two sets."""
    a = n1.intersection(n2)
    e = (len(n1) - len(a)) * (len(n2) - len(a))
    if directed:
        e *= 2
    if self_loops:
        e += len(n1) + len(n2) - len(a)
    return e


def connect_node_sets(
    g,
    s1,
    s2,
    density: Union[float, Tuple[float, float]],
    edge_data: Optional[Dict] = None,
):
    """Connect two node sets at the specified density."""
    if not isinstance(g, nx.Graph):
        raise TypeError(
            "`g` must be a `nx.Graph` subclass, not a `{}`".format(g.__class__)
        )
    if not isinstance(s1, set):
        raise TypeError("node set `s1` must be a `set`, not a `{}`".format(g.__class__))
    if not isinstance(s2, set):
        raise TypeError("node set `s2` must be a `set`, not a `{}`".format(g.__class__))
    m1 = s1.difference(set(g))
    if m1:
        raise ValueError("Nodes missing from graph. " + str(m1))

    m2 = s2.difference(set(g))
    if m2:
        raise ValueError("Nodes missing from graph. " + str(m2))

    edge_data = edge_data or dict()
    existing_edges = set()
    n_possible_edges = _possible_edges(
        s1, s2, directed=nx_is_directed(g), self_loops=False
    )
    for n1, n2 in g.edges():
        if (n1 in s1 and n2 in s2) or (n2 in s1 and n1 in s2):
            existing_edges.add((n1, n2))
            if not nx_is_directed:
                existing_edges.add((n2, n1))
    n_new_edges = max(0, int((n_possible_edges * density) - len(existing_edges)))

    add_new_edges = Fn.compose(
        Fn.map_each(list),
        Fn.apply_each(Fn.shuffle_each()),  # shuffle nodes
        lambda arr: itertools.product(*list(arr)),  # all possible edges
        Fn.filter_each(lambda x: x not in existing_edges),  # only new edges
        Fn.apply_each(lambda x: g.add_edge(x[0], x[1], **edge_data)),  # add edge
        Fn.iter_count(n_new_edges),  # limit number of new edges,
        list,
    )

    add_new_edges((s1, s2))

    return g


def compose_and_connect(
    g: nx.Graph,
    h: nx.Graph,
    density: Union[float, Tuple[float, float]],
    edge_data: Optional[Dict] = None,
) -> nx.Graph:
    """Compose two graphs and connect them at the given density.

    With density being `(existing edges between G, H) / (possible edges
    between G, H)`
    """
    i = nx.compose(g, h)
    density = _resolve_range(density)
    connect_node_sets(i, set(g), set(h), density, edge_data)
    return i


def chain_graph(sequence, graph_class, edge_data: Optional[Dict] = None):
    g = graph_class()
    edge_data = edge_data or dict()
    for n1, n2 in nx.utils.pairwise(sequence):
        g.add_edge(n1, n2, **edge_data)
    return g


_uuid_chain = Fn.compose(
    lambda x: range(x), Fn.map_each(lambda _: str(uuid.uuid4())[-5:])
)


def unique_chain_graph(n: int, graph_class, edge_data: Optional[Dict] = None):
    return chain_graph(_uuid_chain(n), graph_class, edge_data)

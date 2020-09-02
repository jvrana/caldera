from copy import deepcopy as do_deepcopy
from typing import Generator
from typing import Hashable
from typing import Optional
from typing import Type

import networkx as nx


def nx_iter_roots(g: nx.DiGraph) -> Generator[Hashable, None, None]:
    for n in g.nodes():
        if not list(g.predecessors(n)):
            yield n


def nx_iter_leaves(g: nx.DiGraph) -> Generator[Hashable, None, None]:
    for n in g.nodes():
        if not list(g.successors(n)):
            yield n


def nx_copy(g1: nx.Graph, g2: nx.Graph, deepcopy: bool) -> nx.Graph:
    if g2 is None:
        g2 = g1.__class__()
    for n, ndata in g1.nodes(data=True):
        if deepcopy:
            n, ndata = do_deepcopy((n, ndata))
        g2.add_node(n, **ndata)
    for n1, n2, edata in g1.edges(data=True):
        if deepcopy:
            n1, n2, edata = do_deepcopy((n1, n2, edata))
        g2.add_edge(n1, n2, **edata)
    return g2


def nx_shallow_copy(g1: nx.Graph, g2: Optional[nx.Graph] = None) -> nx.Graph:
    return nx_copy(g1, g2, deepcopy=False)


def nx_deep_copy(g1: nx.Graph, g2: Optional[nx.Graph] = None) -> nx.Graph:
    return nx_copy(g1, g2, deepcopy=True)


def nx_class_is_undirected(cls: Type):
    return cls in [nx.Graph, nx.OrderedGraph, nx.MultiGraph, nx.OrderedMultiGraph]


def nx_is_undirected(g: nx.Graph) -> bool:
    return nx_class_is_undirected(g.__class__)


def nx_class_is_directed(cls: Type):
    return cls in [
        nx.DiGraph,
        nx.OrderedDiGraph,
        nx.MultiDiGraph,
        nx.OrderedMultiDiGraph,
    ]


def nx_is_directed(g: nx.Graph) -> bool:
    return nx_class_is_directed(g.__class__)


def nx_copy_to_undirected(
    g: nx.DiGraph, graph_class: Type[nx.Graph] = nx.Graph
) -> Type[nx.Graph]:
    if not nx_class_is_undirected(graph_class):
        raise TypeError(
            "graph_class must be a directed graph type, but found {}".format(
                graph_class
            )
        )
    new_graph = graph_class()
    return nx_deep_copy(g, new_graph)


def nx_to_undirected(
    g: nx.DiGraph, graph_class: Type[nx.Graph] = nx.Graph
) -> Type[nx.Graph]:
    if not nx_class_is_undirected(graph_class):
        raise TypeError(
            "graph_class must be a directed graph type, but found {}".format(
                graph_class
            )
        )
    new_graph = graph_class()
    return nx_shallow_copy(g, new_graph)


def _nx_to_directed(g):
    for n1, n2, edata in g.edges(data=True):
        if not g.has_edge(n2, n1):
            g.add_edge(n2, n1, **edata)


def nx_to_directed(
    g: nx.Graph, graph_class: Type[nx.DiGraph] = nx.DiGraph
) -> Type[nx.Graph]:
    if not nx_class_is_directed(graph_class):
        raise TypeError(
            "graph_class must be a directed graph type, but found {}".format(
                graph_class
            )
        )
    copied = graph_class()
    nx_shallow_copy(g, copied)
    if nx_is_undirected(g):
        _nx_to_directed(copied)
    return copied


def nx_copy_to_directed(
    g: nx.Graph, graph_class: Type[nx.DiGraph] = nx.DiGraph
) -> Type[nx.Graph]:
    if not nx_class_is_directed(graph_class):
        raise TypeError(
            "graph_class must be a directed graph type, but found {}".format(
                graph_class
            )
        )
    copied = graph_class()
    nx_deep_copy(g, copied)
    if nx_is_undirected(copied):
        _nx_to_directed(copied)
    return copied

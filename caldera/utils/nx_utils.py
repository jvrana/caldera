from typing import Generator
from typing import Hashable
from typing import Union
from typing import Type
from typing import overload
from typing import Optional
from copy import deepcopy as do_deepcopy
import networkx as nx

Graph = Union[nx.DiGraph, nx.Graph, nx.MultiGraph, nx.MultiDiGraph, nx.OrderedDiGraph, nx.OrderedGraph]
DirectedGraph = Union[nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph, nx.OrderedMultiDiGraph]
UndirectedGraph = Union[nx.Graph, nx.MultiGraph, nx.OrderedGraph, nx.OrderedMultiGraph]


def iter_roots(g: nx.DiGraph) -> Generator[Hashable, None, None]:
    for n in g.nodes():
        if not list(g.predecessors(n)):
            yield n


def iter_leaves(g: nx.DiGraph) -> Generator[Hashable, None, None]:
    for n in g.nodes():
        if not list(g.successors(n)):
            yield n


def nx_copy(g1: Graph, g2: Graph, deepcopy: bool) -> Graph:
    if g2 is None:
        g2 = g1.__class__()
    for n, ndata in g1.nodes(data=True):
        if deepcopy:
            n, ndata = do_deepcopy((n, ndata))
        g2.add_node(n, **ndata)
    for n1, n2, edata in g2.edges(data=True):
        if deepcopy:
            n1, n2, edata = do_deepcopy((n1, n2, edata))
        g2.add_edge(n1, n2, **edata)


def nx_shallow_copy(g1: Graph, g2: Optional[Graph] = None) -> Graph:
    return nx_copy(g1, g2, deepcopy=False)


def nx_deep_copy(g1: Graph, g2: Optional[Graph] = None) -> Graph:
    return nx_copy(g1, g2, deepcopy=True)


def nx_is_undirected(g: Graph) -> bool:
    if g.__class__ in [nx.Graph, nx.OrderedGraph, nx.MultiGraph, nx.OrderedMultiGraph]:
        return True
    return False


def nx_is_directed(g: Graph) -> bool:
    if g.__class__ in [nx.DiGraph, nx.OrderedDiGraph, nx.MultiDiGraph, nx.OrderedMultiDiGraph]:
        return True
    return False


@overload
def nx_copy_to_undirected(g: ..., graph_class: Type[nx.MultiGraph]) -> Type[nx.MultiGraph]:
    ...


@overload
def nx_copy_to_undirected(g: ..., graph_class: Type[nx.OrderedMultiGraph]) -> Type[nx.OrderedMultiGraph]:
    ...


@overload
def nx_copy_to_undirected(g: ..., graph_class: Type[nx.OrderedGraph]) -> Type[nx.OrderedGraph]:
    ...


def nx_copy_to_undirected(g: nx.DiGraph, graph_class: Type[nx.Graph] = nx.Graph) -> Type[nx.Graph]:
    new_graph = graph_class()
    return nx_deep_copy(g, new_graph)


@overload
def nx_to_undirected(g: ..., graph_class: Type[nx.MultiGraph]) -> Type[nx.MultiGraph]:
    ...


@overload
def nx_to_undirected(g: ..., graph_class: Type[nx.OrderedMultiGraph]) -> Type[nx.OrderedMultiGraph]:
    ...


@overload
def nx_to_undirected(g: ..., graph_class: Type[nx.OrderedGraph]) -> Type[nx.OrderedGraph]:
    ...


def nx_to_undirected(g: nx.DiGraph, graph_class: Type[nx.Graph] = nx.Graph) -> Type[nx.Graph]:
    new_graph = graph_class()
    return nx_shallow_copy(g, new_graph)

import random
from copy import deepcopy as do_deepcopy
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Hashable
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import networkx as nx

from caldera.utils.nx._global_accessor import GraphWithGlobal


T = TypeVar("S")


@overload
def _resolve_range(x: Union[Tuple[float, float], float]) -> float:
    pass


def _rand_float(a: float, b: float):
    return a + (b - a) * random.random()


@overload
def _resolve_range(x: Union[Tuple[float, float], float]) -> float:
    ...


def _resolve_range(x: Union[Tuple[int, int], int]) -> int:
    if isinstance(x, int) or isinstance(x, float):
        return x
    elif isinstance(x, tuple):
        if isinstance(x[0], int):
            return random.randint(*x)
        elif isinstance(x[0], float):
            return _rand_float(*x)
    else:
        raise TypeError


def nx_iter_roots(g: nx.DiGraph) -> Generator[Hashable, None, None]:
    for n in g.nodes():
        if not list(g.predecessors(n)):
            yield n


def nx_iter_leaves(g: nx.DiGraph) -> Generator[Hashable, None, None]:
    for n in g.nodes():
        if not list(g.successors(n)):
            yield n


NodeGenerator = Generator[Tuple[T, Dict], None, None]
EdgeGenerator = Generator[Tuple[T, T, Dict], None, None]


def nx_copy(
    from_graph: nx.Graph,
    to_graph: Union[nx.Graph, Type[nx.Graph]],
    *,
    node_transform: Optional[Callable[[NodeGenerator], NodeGenerator]] = None,
    edge_transform: Optional[Callable[[EdgeGenerator], EdgeGenerator]] = None,
    deepcopy: bool = False
) -> nx.Graph:
    """Copies node, edges, node_data, and edge_data from graph `g1` to graph
    `g2`. If `g2` is None, a new graph of type `g1.__class__` is created. If
    `g2` is a class or subclass of `nx.Graph`, a new graph of that type is
    created.

    If `deepcopy` is set to True, the copy will perform a deepcopy of all node
    and edge data. If false, only shallow copies will be created.

    `node_transform` and `edge_transform` can be provided to perform a transform
    on the on `g1.nodes(data=True)` and `g1.edges(data=True)` iterators
    during copy. The node transform should return a `Generator[Tuple[T, dict], None, None]`
    while the edge transform should return a `Generator[Tuple[T, T, dict], None, None]`.
    These transforms may include skipping certain nodes or edges, transforming the node or edge
    data, or transforming the node keys themselves.

    Some example transforms include:

    .. code-block:: python

        def node_to_str(gen):
            for n, ndata in gen:
                yield (str(n), ndata)

        def remove_self_loops(gen):
            for n1, n2, edata in gen:
                if n1 == n2:
                    yield (n1, n2, edata)

        nx_copy(g1, None, node_transform=node_to_str, edge_transform=remove_self_loops, deepcopy=True)


    :param from_graph: graph to copy from
    :param to_graph: graph to copy to
    :param node_transform: optional transform applied to the `from_graph.nodes(data=True)` iterator
    :param edge_transform: optional transform applied to the `from_graph.edges(data=True)` iterator
    :param literal_transform: if True, node_transform will *not* be applied following the edge_transform.
        This may result in unintentional edges being created.
    :param deepcopy:
    :return:
    """
    if to_graph is None:
        to_graph = from_graph.__class__()
    elif isinstance(to_graph, type) and issubclass(to_graph, nx.Graph.__class__):
        to_graph = to_graph()

    node_iter = from_graph.nodes(data=True)
    if node_transform:
        node_iter = node_transform(node_iter)

    for n, ndata in node_iter:
        if deepcopy:
            n, ndata = do_deepcopy((n, ndata))
        to_graph.add_node(n, **ndata)

    edge_iter = to_graph.edges(data=True)
    if edge_transform:
        edge_iter = edge_transform(edge_iter)

    for n1, n2, edata in edge_iter:
        if deepcopy:
            n1, n2, edata = do_deepcopy((n1, n2, edata))
        to_graph.add_edge(n1, n2, **edata)

    if hasattr(from_graph, GraphWithGlobal.get_global.__name__) and hasattr(
        to_graph, GraphWithGlobal.get_global.__name__
    ):
        gdata = from_graph.get_global()
        if deepcopy:
            gdata = do_deepcopy(gdata)
        else:
            gdata = dict(gdata)
        to_graph.set_global(gdata)
    return to_graph


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

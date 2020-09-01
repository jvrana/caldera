import random
from typing import Callable
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Union

import networkx as nx
import torch

from caldera.utils.functional import Functional as Fn
from caldera.utils.nx.convert import add_default


def nx_random_features(g: nx.DiGraph, n_feat: int, e_feat: int, g_feat: int):
    for _, ndata in g.nodes(data=True):
        ndata["features"] = torch.randn(n_feat)
    for _, _, edata in g.edges(data=True):
        edata["features"] = torch.randn(e_feat)
    g.data = {"features": torch.randn(g_feat)}
    return g


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


def feature_info(g, global_key: str = None):
    collect_values_by_key = Fn.compose(
        Fn.map_each(lambda x: x.items()),
        Fn.chain_each(),
        Fn.group_each_by_key(lambda x: x[0]),
        Fn.map_each(lambda x: (x[0], [_x[1] for _x in x[1]])),
        dict,
    )  # from a list of dictionaries, collect values by key

    unique_value_types = Fn.compose(
        Fn.index_each(-1),
        collect_values_by_key,
        lambda x: {k: {_v.__class__ for _v in v} for k, v in x.items()},
    )

    unique_value_types(g.nodes(data=True))

    return {
        "node": {"keys": unique_value_types(g.nodes(data=True))},
        "edge": {"keys": unique_value_types(g.edges(data=True))},
        "global": {"keys": unique_value_types([(g.get_global(global_key),)])},
    }


def random_node(g, n=1):
    nodes = [random.choice(list(g.nodes)) for _ in range(n)]
    if n == 1:
        nodes = nodes[0]
    return nodes


def random_edge(g, n=1):
    edges = [random.choice(list(g.edges)) for _ in range(n)]
    if n == 1:
        edges = edges[0]
    return edges


def annotate_shortest_path(
    g: nx.Graph,
    annotate_nodes: bool = True,
    annotate_edges: bool = True,
    source_key: str = "source",
    target_key: str = "target",
    path_key: str = "shortest_path",
) -> nx.Graph:
    if not annotate_edges and not annotate_nodes:
        raise ValueError("Must annotate either nodes or edges (or both)")
    source, target = random_node(g, 2)

    g.nodes[source][source_key] = True
    g.nodes[target][target_key] = True

    try:
        path = nx.shortest_path(g, source=source, target=target)
    except nx.NetworkXNoPath:
        path = []

    add_default(g, node_data={target_key: False, source_key: False, path_key: False})
    add_default(g, edge_data={path_key: False})

    if annotate_nodes:
        for n in path:
            g.nodes[n][path_key] = True

    if annotate_edges:
        for n1, n2 in nx.utils.pairwise(path):
            g.edges[(n1, n2)][path_key] = True

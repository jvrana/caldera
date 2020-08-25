from .path_utils import PathSymbol, PathNpSum
from typing import Hashable, Union, Tuple, List, Optional, Callable, Dict, overload
from heapq import heappop
from heapq import heappush
from itertools import count
import numpy as np
import networkx as nx
from caldera.utils.nx.graph_utils import Graph


def _weight_function(e, k):
    if e is None:
        return None
    return e[k]


def _multisource_dijkstra(
    g: Graph,
    sources: Union[Tuple[Hashable, ...], List[Hashable]],
    symbols: Tuple[Union[str, PathSymbol], ...],
    func: Callable = lambda x: x,
    target: Optional[Union[Tuple[Hashable, ...], List[Hashable]]] = None,
    accumulators=None,
    init=None,
    cutoff=None,
    paths=None,
    pred=None,
) -> Dict[Hashable, Union[float, int]]:

    _symbols = symbols[:]
    symbols = []
    for s in _symbols:
        if isinstance(s, str):
            s = PathSymbol(s, PathNpSum)
        symbols.append(s)

    accumulators = accumulators or {}
    init = init or {}

    # successor dictionary
    g_succ = g._succ if g.is_directed() else g._adj

    # push/pop methods to use
    push = heappush
    pop = heappop

    # node to shortest distance, breakdown
    dist_parts = {}

    # node to shortest distance
    dist = {}

    # node to shortest distance seen
    seen = {}
    c = count()
    fringe = []

    # initial/default values for each symbol

    for sym in symbols:
        if accumulators.get(sym, "sum") == "sum":
            init.setdefault(sym, 0.0)
        elif accumulators.get(sym, "product"):
            init.setdefault(sym, 1.0)
        else:
            raise ValueError("Accumulator '{}' not recognized".format(sym))
    init = np.array([s.iden_fill for s in symbols])

    # push the initial values for the sources
    for source in sources:
        if source not in g:
            raise nx.NodeNotFound("Source {} is not in G".format(source))
        seen[source] = func(*init)
        push(fringe, (0, next(c), source, init))

    # modified dijkstra's
    while fringe:
        (_, _, v, d) = pop(fringe)
        # d np.array of values
        if v in dist_parts:
            continue  # already searched this node
        dist_parts[v] = d
        dist[v] = func(*d)
        if v == target:
            break
        for u, e in g_succ[v].items():
            # vu cost breakdown for  each symbol
            costs = np.array([_weight_function(e, sym.name) for sym in symbols])

            # vu_dist break down using accumulating function
            x = np.stack([dist_parts[v], costs])
            vu_dist_parts = []
            for i, _x in enumerate(x.T):
                vu_dist_parts.append(symbols[i].accumulator(_x))
            vu_dist_parts = np.array(vu_dist_parts)
            vu_dist = func(*vu_dist_parts)

            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist_parts:
                if vu_dist < dist[u]:
                    raise ValueError("Contradictory paths found:", "negative weights?")
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = func(*vu_dist_parts)
                push(fringe, (vu_dist, next(c), u, vu_dist_parts))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = v
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)
                    fringe = []
    return dist


@overload
def multisource_dijkstras(
    g: ...,
    sources: ...,
    symbols: ...,
    f: ...,
    target: Optional = None,
    cutoff: Optional = None,
) -> Union[Tuple[Dict[Hashable, Union[float, int]], Dict[Hashable, List[Hashable]]]]:
    ...


def multisource_dijkstras(
    g: Graph,
    sources: Union[Tuple[Hashable, ...], List[Hashable]],
    symbols: Tuple[Union[str, PathSymbol], ...],
    func: Callable = lambda x: x,
    target: Hashable = None,
    cutoff: Optional[Union[float, int]] = None,
) -> Union[
    Tuple[Union[float, int], List[Hashable]],
]:
    if sources is None or not len(sources):
        raise ValueError("sources must not be empty")
    if target in sources:
        return (0, [target])
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = _multisource_dijkstra(
        g=g,
        sources=sources,
        func=func,
        target=target,
        symbols=symbols,
        paths=paths,
        cutoff=cutoff,
    )
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError:
        raise nx.NetworkXNoPath("No path to {}.".format(target))

from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Union

import torch

from ._floyd_warshall import floyd_warshall
from ._utils import get_edge_dict
from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils import long_isin
from caldera.utils import torch_scatter_group


@overload
def tensor_induce(data: ..., nodes: torch.BoolTensor, k: ...) -> torch.BoolTensor:
    ...


def tensor_induce(
    data: Union[GraphData, GraphBatch], nodes: torch.LongTensor, k: int
) -> torch.LongTensor:
    if isinstance(nodes, int):
        nodes = torch.LongTensor([nodes])
    elif nodes.dtype == torch.long and nodes.ndim == 0:
        nodes = nodes.expand(1)

    visited = nodes.detach().clone()
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


@overload
def induce(data: ..., nodes: torch.BoolTensor, k: ...) -> torch.BoolTensor:
    ...


def induce(
    data: Union[GraphData, GraphBatch],
    nodes: torch.LongTensor,
    k: int,
    edge_dict: Optional[Dict] = None,
) -> torch.LongTensor:
    assert nodes.ndim == 1
    if nodes.dtype == torch.long:
        visited = bfs_nodes(nodes, data.edges, depth=k, edge_dict=edge_dict)
        ret = torch.tensor(list(visited), dtype=torch.long)
        return ret
    elif nodes.dtype == torch.bool:
        nidx = torch.where(nodes)[0]
        visited = bfs_nodes(nidx, data.edges, depth=k, edge_dict=edge_dict)
        ret = torch.tensor([False] * data.num_nodes)
        ret[torch.LongTensor(list(visited))] = True
        return ret
    else:
        raise ValueError("{} is not a valid type".format(data.dtype))


@overload
def neighbors(data: ..., nodes: torch.BoolTensor) -> torch.BoolTensor:
    ...


# TODO: dispatch method with multidim neighbors and floyd-warshall
def neighbors(
    data: Union[GraphData, GraphBatch],
    nodes: torch.LongTensor,
    reverse: bool = False,
    undirected: bool = False,
) -> torch.LongTensor:
    """Return the neighbors of the provided nodes.

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

    if undirected:
        reachable1 = long_isin(data.edges[0], nodes)
        dest1 = data.edges[1][reachable1]
        reachable2 = long_isin(data.edges[1], nodes)
        dest2 = data.edges[0][reachable2]
        dest = torch.unique(torch.cat([dest1, dest2]))
    else:
        if reverse:
            i, j = 1, 0
        else:
            i, j = 0, 1
        reachable = long_isin(data.edges[i], nodes)
        dest = data.edges[j][reachable]

    if is_bool:
        ret = torch.full((data.num_nodes,), False, dtype=torch.bool)
        ret[dest] = True
    else:
        ret = torch.unique(dest, sorted=True)
    return ret


def bfs_nodes(
    src: Union[int, List[int], Tuple[int, ...], torch.LongTensor],
    edges: torch.LongTensor,
    depth: Optional[int] = None,
    edge_dict: Optional[Dict] = None,
) -> Set[Hashable]:
    """Return nodes from a breadth-first search. Optionally provide a depth.

    :param src:
    :param edges:
    :param depth:
    :return:
    """
    if edge_dict is None:
        edge_dict = get_edge_dict(edges)
    if torch.is_tensor(src):
        nlist = src.tolist()
    elif isinstance(src, list):
        nlist = src[:]
    elif isinstance(src, tuple):
        nlist = list(src)
    elif isinstance(src, int):
        nlist = [src]

    to_visit = nlist[:]
    depths = [0] * len(nlist)
    discovered = set()

    i = 0
    while to_visit and (depth is None or i < depth):
        v = to_visit.pop(0)
        d = depths.pop(0)
        if depth is not None and d > depth:
            continue

        discovered.add(v)
        if depth is None or d + 1 <= depth:
            if v in edge_dict:
                neighbors = edge_dict[v]
                for n in neighbors:
                    if n not in discovered:
                        to_visit.append(n)
                        depths.append(d + 1)
    return discovered


def _fw_neighbors(m, n, depth):
    x = m[n]
    reachable = torch.logical_and(x != float("inf"), x <= depth)
    if n.dtype == torch.bool:
        return reachable
    idx = torch.where(reachable)
    return idx


@overload
def floyd_warshall_neighbors(
    data: ...,
    nodes: Tuple[torch.LongTensor, ...],
    depth: ...,
    matrix: ...,
    return_matrix: ...,
) -> Union[torch.LongTensor, Tuple[torch.LongTensor, torch.FloatTensor]]:
    ...


@overload
def floyd_warshall_neighbors(
    data: ...,
    nodes: Tuple[torch.BoolTensor, ...],
    depth: ...,
    matrix: ...,
    return_matrix: ...,
) -> Union[torch.BoolTensor, Tuple[torch.BoolTensor, torch.FloatTensor]]:
    ...


def floyd_warshall_neighbors(
    data: Union[GraphData, GraphBatch],
    nodes: Union[torch.LongTensor, int],
    depth: int = 1,
    matrix: Optional[torch.FloatTensor] = None,
    return_matrix: bool = False,
) -> Union[torch.LongTensor, Tuple[torch.LongTensor, torch.FloatTensor]]:
    if matrix is None:
        matrix = floyd_warshall(data)

    if isinstance(nodes, int):
        nodes = torch.LongTensor(nodes)

    if not isinstance(nodes, tuple) and not nodes.ndim <= 2:
        raise ValueError("`nodes` must be a 0, 1, or 2-dimensional tensor")

    if isinstance(nodes, tuple):
        ret = []
        for n in nodes:
            ret.append(floyd_warshall_neighbors(data, n, depth=depth, matrix=matrix))
        ret = tuple(ret)
    else:
        idx = _fw_neighbors(matrix, nodes, depth)
        if torch.is_tensor(nodes) and nodes.ndim == 2:
            ret = torch_scatter_group(idx[-1], idx[0])
        else:
            ret = idx[-1]

    if return_matrix:
        ret = (ret, matrix)
    return ret

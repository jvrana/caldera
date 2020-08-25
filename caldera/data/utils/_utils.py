from typing import Callable
from typing import Dict
from typing import Hashable
from typing import List
from typing import Set
from typing import Tuple

import torch


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
    return s3


# @overload
# def add_edges(data: GraphBatch, fill_value: ..., kind: ...) -> GraphBatch:
#     """
#     Adds edges to the :class:`caldera.data.GraphBatch` instance.
#     """
#     ...


def get_edge_dict(edges: torch.LongTensor) -> Dict[Hashable, Set[Hashable]]:
    src, dest = edges.tolist()
    edge_dict = {}
    for _src, _dest in zip(src, dest):
        edge_dict.setdefault(_src, set())
        edge_dict[_src].add(_dest)
    return edge_dict

# optimized methods for grouping tensors
import torch
from typing import Dict, List, Tuple
import itertools


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


@torch.jit.script
def unique_with_counts(idx, grouped: Dict[int, int]):
    for x in idx:
        if x.item() not in grouped:
            grouped[x.item()] = 1
        else:
            grouped[x.item()] += 1

    counts = torch.zeros(len(grouped), dtype=torch.long)
    values = torch.empty(len(grouped), dtype=idx.dtype)
    for i, (k, v) in enumerate(grouped.items()):
        values[i] = k
        counts[i] = v
    a = torch.argsort(values)

    return values[a], counts[a]


@torch.jit.script
def _jit_scatter_group(x: torch.Tensor, idx: torch.Tensor, d: Dict[int, int]) -> Tuple[
    torch.Tensor, List[torch.Tensor]]:
    x = x[torch.argsort(idx)]
    groups, b = unique_with_counts(idx, d)
    i_a = 0
    arr_list = []
    for i_b in b:
        arr_list.append(x[i_a:i_a + i_b.item()])
        i_a += i_b.item()
    return groups, arr_list


def scatter_group(x: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    return _jit_scatter_group(x, idx, {})
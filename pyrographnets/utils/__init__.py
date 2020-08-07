import itertools
from pyrographnets.utils.jit import scatter_group, stable_arg_sort_long, jit_scatter_group, unique_with_counts
from pyrographnets.utils.torch_utils import same_storage
from typing import TypeVar
from typing import Callable
from typing import List, Dict


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _first(i):
    """Select the first element in an iterable"""
    return next((x for x in itertools.tee(i)[0]))


def dict_collate(d1: Dict[K, T], d2: Dict[K, T], collate_fn: Callable[[List[T]], V]) -> Dict[K, V]:
    d = {}
    for k, v in d1.items():
        if k not in d:
            d[k] = [v]
        else:
            d[k].append(v)
    for k, v in d2.items():
        if k not in d:
            d[k] = [v]
        else:
            d[k].append(v)
    return {k: collate_fn(v) for k, v in d.items()}
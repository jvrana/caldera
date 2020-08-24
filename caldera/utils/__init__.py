import itertools
from typing import Callable
from typing import Dict
from typing import List
from typing import TypeVar

from caldera.utils.jit import torch_scatter_group
from caldera.utils.jit import stable_arg_sort_long
from caldera.utils.torch_utils import deterministic_seed
from caldera.utils.torch_utils import same_storage
from caldera.utils.jit import long_isin
from caldera.utils.torch_utils import tensor_is_empty, reindex_tensor, unravel_index

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _first(i):
    """Select the first element in an iterable."""
    return next(x for x in itertools.tee(i)[0])


def dict_collate(
    d1: Dict[K, T], d2: Dict[K, T], collate_fn: Callable[[List[T]], V]
) -> Dict[K, V]:
    """
    Apply a collation function to a pair dictionaries.
    """
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

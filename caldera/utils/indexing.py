import operator
from functools import reduce
from typing import Generator
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union

import torch

SizeType = Union[torch.Size, Tuple[int, ...]]


def repeat_roll(shape: SizeType, dim: int) -> torch.Tensor:
    """Roll over shape, rolling the repeat over different dimension."""
    torch.arange(shape[dim])

    repeat_dims = list(shape)[:dim]
    interleave_dims = list(shape[dim + 1 :])

    def prod(x):
        if not x:
            return 1
        else:
            return reduce(operator.mul, x)

    n = prod(interleave_dims)
    m = prod(repeat_dims)
    return torch.repeat_interleave(torch.arange(shape[dim]), n).repeat(m)


def unroll_index(
    shape: SizeType, dtype: Type = torch.long
) -> Tuple[torch.LongTensor, ...]:
    idx = tuple()
    for dim in range(len(shape)):
        idx = idx + (repeat_roll(shape, dim).to(dtype),)
    return idx


@overload
def unravel_index(index: int, shape: ...) -> Tuple[int]:
    ...


def unravel_index(
    index: torch.LongTensor, shape: SizeType
) -> Tuple[torch.LongTensor, ...]:
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


@overload
def reindex_tensor(a: torch.Tensor) -> torch.Tensor:
    ...


def reindex_tensor(
    a: torch.Tensor, *tensors: Tuple[torch.Tensor, ...]
) -> Tuple[torch.tensor, ...]:
    """Reindex a tensor to lowest index. Handles multiple tensors and tensors
    with many dimensions.

    .. code-block:: python

        a = torch.tensor([1, 1, 1, 4, 0, 5, 0, 0, 0])
        b = reindex(a)
        print(b)
        # >> tensor([0, 0, 0, 1, 2, 3, 2, 2, 2])

    .. code-block:: pythong

        # multiple tensors with multiple dimensions
        a = torch.tensor([1, 1, 1, 1, 0, 2, 0, 5, 6])
        b = torch.tensor([[6, 5, 1, 70], [0, 80, 5, 6]])
        expected1 = torch.tensor([0, 0, 0, 0, 1, 2, 1, 3, 4])
        expected2 = torch.tensor([[4, 3, 0, 5], [1, 6, 3, 4]])
        c, d = reindex_tensor(a, b)
        assert torch.all(c == expected1)
        assert torch.all(d == expected2)


    :param a: tensor to reindex
    :return: new reindexed tensor
    """

    values_list = []
    all_tensors = [a] + list(tensors)
    for t in all_tensors:
        if not t.dtype == a.dtype:
            raise ValueError(
                "All tensors must be same type. {} != {}".format(t.dtype, a.dtype)
            )
        values_list.append(t.flatten().tolist())

    new_tensors = []

    replace = {}
    j = 0
    for tidx, tlist in enumerate(values_list):
        t = all_tensors[tidx]
        b = torch.empty_like(t)
        new_tensors.append(b)
        for i, _t in enumerate(tlist):
            if _t not in replace:
                replace[_t] = j
                j += 1
            bidx = unravel_index(i, t.shape)
            b[bidx] = replace[_t]

    if tensors:
        return new_tensors
    else:
        return new_tensors[0]

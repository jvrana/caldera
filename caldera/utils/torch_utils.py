import random
from typing import overload

import numpy
import torch
from typing import Tuple
from typing import Union


def empty(x: torch.Tensor) -> bool:
    """
    Return whether the tensor is empty.
    """
    return 0 in x.shape


def same_storage(
    x: torch.Tensor, y: torch.Tensor, empty_does_not_share_storage: bool = True
) -> bool:
    """
    Checks if two tensors share storage.

    :param x: first tensor
    :param y: second tensor
    :param empty_does_not_share_storage: if True (default), will return False if
        either tensor is empty (despite that they technically data_ptr are the same).
    :return: if the tensor shares the same storage
    """
    if empty_does_not_share_storage and (empty(x) or empty(y)):
        return False
    x_ptrs = {e.data_ptr() for e in x.view(-1)}
    y_ptrs = {e.data_ptr() for e in y.view(-1)}
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


# TODO: add more options for deterministic_seed?
def deterministic_seed(seed: int, cudnn_deterministic: bool = False):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@overload
def to_one_hot(arr: numpy.ndarray, mx: int) -> numpy.ndarray:
    ...


def to_one_hot(arr: torch.tensor, mx: int) -> torch.tensor:
    if torch.is_tensor(arr):

        oh = torch.zeros((arr.shape[0], mx))
    else:
        oh = numpy.zeros((arr.shape[0], mx))
    for i, a in enumerate(arr):
        oh[i, a] = 1.0
    return oh


@overload
def unravel_index(index: int, shape: ...) -> Tuple[int]:
    ...


def unravel_index(
    index: torch.LongTensor, shape: Union[Tuple[int, ...], torch.Size]
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
    """
    Reindex a tensor to lowest index. Handles multiple tensors and tensors with
    many dimensions.

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

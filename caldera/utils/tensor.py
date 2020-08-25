import random
from typing import overload, Union
from typing import List
from typing import Tuple
import numpy
import torch


def tensor_is_empty(x: torch.Tensor) -> bool:
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
    if empty_does_not_share_storage and (tensor_is_empty(x) or tensor_is_empty(y)):
        return False
    x_ptrs = {e.data_ptr() for e in x.view(-1)}
    y_ptrs = {e.data_ptr() for e in y.view(-1)}
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


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


@torch.jit.script
def stable_arg_sort(arr, mn: float, mx: float):
    dim = -1
    if not dim == -1:
        raise ValueError("only last dimension sort is supported. Try reshaping tensor.")
    delta_shape = list(arr.shape)
    delta_shape[dim] = 1
    delta = torch.linspace(mn, mx, arr.shape[dim])
    delta = delta.repeat(delta_shape)
    return torch.argsort(arr + delta, dim=dim)


@torch.jit.script
def stable_arg_sort_long(arr):
    """Stable sort of long tensors.

    Note that Pytorch 1.5.0 does not have a stable sort implementation.
    Here we simply add a delta value between 0 and 1 (exclusive) and
    assuming we are using integers, call torch.argsort to get a stable
    sort.
    """
    dim = -1
    if not (arr.dtype == torch.long or arr.dtype == torch.int):
        raise ValueError("only torch.Long or torch.Int allowed")
    return stable_arg_sort(arr, 0.0, 0.99)


def torch_scatter_group(
    x: torch.Tensor, idx: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Group a tensor by indices. This is equivalent to successive applications
    of `x[torch.where(x == index)]` for all provided sorted indices.

    Example:

    .. code-block:: python

        idx = torch.tensor([2, 2, 0, 1, 1, 1, 2])
        x = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        uniq_sorted_idx, out = scatter_group(x, idx)

        # node the idx is sorted
        assert torch.all(torch.eq(out[0], torch.tensor([0, 1, 2])))

        # where idx == 0
        assert torch.all(torch.eq(out[1][0], torch.tensor([2])))

        # where idx == 1
        assert torch.all(torch.eq(out[1][1], torch.tensor([3, 4, 5])))

        # where idx == 2
        assert torch.all(torch.eq(out[1][2], torch.tensor([0, 1, 6])))

    :param x: tensor to group
    :param idx: indices
    :return: tuple of unique, sorted indices and a list of tensors corresponding to the groups
    """
    arg = stable_arg_sort_long(idx)
    x = x[arg]
    groups, b = torch.unique(idx, return_counts=True)
    i_a = 0
    arr_list = []
    for i_b in b:
        arr_list.append(x[i_a : i_a + i_b.item()])
        i_a += i_b.item()
    return groups, arr_list


def long_isin(ar1, ar2, assume_unique: bool = False, invert: bool = False):
    dim = -1
    if ar1.dtype != torch.long or ar2.dtype != torch.long:
        raise ValueError("Arrays be torch.LongTensor")
    if ar2.ndim > 1:
        raise ValueError(
            "Unable to broadcast shape {}. Second tensor must be a "
            "1-dimensional.".format(ar2.shape)
        )

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = torch.unique(ar1, return_inverse=True)
        ar2 = torch.unique(ar2, dim=None)
        # TODO: how to handle repeats and unique in multidimensional tensor?

    # if ar2.ndim > 1:
    #     s = list(ar2.shape)
    #     s[dim] = 1
    #     ar1 = ar1.repeat(s)
    ar = torch.cat((ar1, ar2), axis=dim)

    # We need this to be a stable sort
    order = stable_arg_sort_long(ar)
    sar = torch.gather(ar, dim, order)
    if invert:
        bool_ar = sar[1:] != sar[:-1]
    else:
        bool_ar = sar[1:] == sar[:-1]
    flag = torch.cat((bool_ar, torch.tensor([invert])))
    ret = torch.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[: len(ar1)]
    else:
        return ret[rev_idx]
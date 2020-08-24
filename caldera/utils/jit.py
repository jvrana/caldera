from typing import Dict
from typing import List
from typing import Tuple

import torch


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
    if not dim == -1:
        raise ValueError("only last dimension sort is supported. Try reshaping tensor.")
    delta_shape = list(arr.shape)
    delta_shape[dim] = 1
    delta = torch.linspace(0, 0.99, arr.shape[dim])
    delta = delta.repeat(delta_shape)
    return torch.argsort(arr + delta, dim=dim)


# @torch.jit.script
# def torch_unique(
#     arr: torch.Tensor, return_counts: bool = False, return_inverse: bool = False
# ):
#     """A much faster implementation (300X) of `torch.unique` for 1-D tensors.
#      Approximately equal to `torch.unique(arr)`
#
#      .. code-block:: pythong
#
#         unique(arr, {}, {}, return_counts=True, return_inverse=True)
#     """
#     grouped: Dict[int, int] = {}
#     idx: Dict[int, int] = {}
#     for x in arr:
#         if x.item() not in grouped:
#             grouped[x.item()] = 1
#         else:
#             grouped[x.item()] += 1
#
#     values = torch.empty(len(grouped), dtype=arr.dtype)
#     if return_inverse:
#         idx_len = len(arr)
#     else:
#         idx_len = 0
#     idxs = torch.empty(idx_len, dtype=torch.long)
#
#     if return_counts:
#         counts_len = len(grouped)
#     else:
#         counts_len = 0
#     counts = torch.zeros(counts_len, dtype=torch.long)
#     for i, (k, v) in enumerate(grouped.items()):
#         values[i] = k
#         if return_counts:
#             counts[i] = v
#
#     a = torch.argsort(values)
#     values = values[a]
#     if return_inverse:
#         for i, v in enumerate(values):
#             idx[v.item()] = i
#
#         for i, _a in enumerate(arr):
#             idxs[i] = idx[_a.item()]
#
#     if return_counts:
#         counts = counts[a]
#
#     return values, idxs, counts


# @torch.jit.script
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

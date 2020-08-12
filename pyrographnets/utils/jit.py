import torch
from typing import Dict, List, Tuple


@torch.jit.script
def stable_arg_sort_long(arr):
    """Stable sort of long tensors.

    Note that Pytorch 1.5.0 does not have a stable sort implementation.
    Here we simply add a delta value between 0 and 1 (exclusive) and
    assuming we are using integers, call
    torch.argsort to get a stable sort."""
    delta = torch.linspace(0, 0.99, arr.shape[0])
    return torch.argsort(arr + delta)


@torch.jit.script
def unique_with_counts(arr: torch.Tensor, grouped: Dict[int, int]):
    """
    Equivalent to `np.unqiue(x, return_counts=True)`
    
    :param arr:
    :param grouped:
    :return:
    """
    for x in arr:
        if x.item() not in grouped:
            grouped[x.item()] = 1
        else:
            grouped[x.item()] += 1

    counts = torch.zeros(len(grouped), dtype=torch.long)
    values = torch.empty(len(grouped), dtype=arr.dtype)
    for i, (k, v) in enumerate(grouped.items()):
        values[i] = k
        counts[i] = v
    a = torch.argsort(values)

    return values[a], counts[a]


@torch.jit.script
def jit_scatter_group(
    x: torch.Tensor, idx: torch.Tensor, d: Dict[int, int]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Assume idx is a sorted index

    :param x:
    :param idx:
    :param d:
    :return:
    """
    arg = stable_arg_sort_long(idx)
    x = x[arg]
    groups, b = unique_with_counts(idx, d)
    i_a = 0
    arr_list = []
    for i_b in b:
        arr_list.append(x[i_a : i_a + i_b.item()])
        i_a += i_b.item()
    return groups, arr_list


def scatter_group(
    x: torch.Tensor, idx: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Group a tensor by indices. This is equivalent to successive applications of `x[torch.where(x == index)]`
    for all provided sorted indices

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
    return jit_scatter_group(x, idx, {})

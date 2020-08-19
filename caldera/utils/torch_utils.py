import random
from typing import overload

import numpy
import torch


def empty(x: torch.Tensor) -> bool:
    return 0 in x.shape


def same_storage(x: torch.Tensor, y: torch.Tensor,
                 empty_does_not_share_storage: bool = True) -> bool:
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

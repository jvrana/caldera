import torch
import random
import numpy
from typing import overload


def same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Checks if two tensors share storage."""
    print(x)
    print(y)
    print(x.shape)
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


# TODO: add more options for deterministic_seed?
def deterministic_seed(seed: int, cudnn_deterministic: bool = False):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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



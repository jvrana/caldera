from typing import NamedTuple

import torch

# TODO: why is GraphTuple have variables in a different order from everything else?
class GraphTuple(NamedTuple):
    e: torch.Tensor
    x: torch.Tensor
    g: torch.Tensor

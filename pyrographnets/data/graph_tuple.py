from typing import NamedTuple

import torch


class GraphTuple(NamedTuple):
    e: torch.Tensor
    x: torch.Tensor
    g: torch.Tensor

"""_random_node_mask.py.

Apply random node mask to the graph.
"""
from typing import overload

import torch
from torch import distributions

from ._base import TransformBase
from caldera.data import GraphBatch
from caldera.data import GraphData


class RandomNodeMask(TransformBase):
    def __init__(self, dropout: float):
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("Edge dropout rate must be between [0., 1.]")
        self.dist = distributions.Bernoulli(1.0 - dropout)

    @overload
    def __call__(self, data: GraphData) -> GraphData:
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        mask = self.dist.sample((data.num_nodes,)).to(torch.bool)
        return data.apply_node_mask(mask)

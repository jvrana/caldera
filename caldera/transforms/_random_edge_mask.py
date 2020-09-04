"""_random_edge_mask.py.

Apply random edge mask to the graph.
"""
from typing import overload

import torch
from torch import distributions

from ._base import TransformBase
from caldera.data import GraphBatch
from caldera.data import GraphData


class RandomEdgeMask(TransformBase):
    def __init__(self, dropout: float):
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("Edge dropout rate must be between [0., 1.]")
        self.dist = distributions.Bernoulli(1.0 - dropout)

    @overload
    def transform_each(self, data: GraphData) -> GraphData:
        ...

    def transform(self, data: GraphBatch) -> GraphBatch:
        mask = self.dist.sample((data.num_edges,)).to(torch.bool)
        return data.apply_edge_mask(mask)

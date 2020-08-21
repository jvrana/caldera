"""
random_node_mask.py

Apply random node mask to the graph.
"""
from .base import TransformBase
from caldera.data import GraphData, GraphBatch
from typing import overload
from torch import distributions
import torch


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

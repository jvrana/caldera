r"""
Blocks (:mod:`caldera.blocks`)
==============================

.. currentmodule:: caldera.blocks

This module provides representations of Molecules and Molecular Assemblies.

.. autosummary::
    :toctree: generated/

    Aggregator
"""
from caldera.blocks.aggregator import Aggregator
from caldera.blocks.aggregator import MultiAggregator
from caldera.blocks.edge_block import AggregatingEdgeBlock
from caldera.blocks.edge_block import EdgeBlock
from caldera.blocks.flex import Flex
from caldera.blocks.global_block import AggregatingGlobalBlock
from caldera.blocks.global_block import GlobalBlock
from caldera.blocks.mlp import MLP
from caldera.blocks.node_block import AggregatingNodeBlock
from caldera.blocks.node_block import NodeBlock

__all__ = [
    Aggregator,
    MultiAggregator,
    AggregatingEdgeBlock,
    EdgeBlock,
    Flex,
    AggregatingNodeBlock,
    AggregatingEdgeBlock,
    AggregatingGlobalBlock,
    NodeBlock,
    GlobalBlock,
    MLP,
]

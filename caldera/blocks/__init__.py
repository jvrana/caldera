r"""
.. currentmodule:: caldera.blocks

Network building blocks for creating graph neural networks.

Generic Blocks
--------------

.. autosummary::
    :toctree: generated/

    Flex
    Dense

Encoder/Decoder Blocks
----------------------

.. autosummary::
    :toctree: generated/

    NodeBlock
    EdgeBlock
    GlobalBlock

Message Passing Blocks
----------------------

.. autosummary::
    :toctree: generated/

    AggregatingNodeBlock
    AggregatingEdgeBlock
    AggregatingGlobalBlock
    Aggregator
    MultiAggregator
"""
from caldera.blocks.aggregator import Aggregator
from caldera.blocks.aggregator import MultiAggregator
from caldera.blocks.dense import Dense
from caldera.blocks.edge_block import AggregatingEdgeBlock
from caldera.blocks.edge_block import EdgeBlock
from caldera.blocks.flex import Flex
from caldera.blocks.global_block import AggregatingGlobalBlock
from caldera.blocks.global_block import GlobalBlock
from caldera.blocks.node_block import AggregatingNodeBlock
from caldera.blocks.node_block import NodeBlock

__all__ = [
    "Aggregator",
    "MultiAggregator",
    "AggregatingEdgeBlock",
    "EdgeBlock",
    "Flex",
    "AggregatingNodeBlock",
    "AggregatingEdgeBlock",
    "AggregatingGlobalBlock",
    "NodeBlock",
    "GlobalBlock",
    "Dense",
]

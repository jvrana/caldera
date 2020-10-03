r"""
Network building blocks for creating graph neural networks.

Generic Blocks
--------------

.. autosummary::
    :toctree: _generated/

    Flex
    Dense

Encoder/Decoder Blocks
----------------------

.. autosummary::
    :toctree: _generated/

    NodeBlock
    EdgeBlock
    GlobalBlock

Message Passing Blocks
----------------------

.. autosummary::
    :toctree: _generated/

    AggregatingNodeBlock
    AggregatingEdgeBlock
    AggregatingGlobalBlock
    Aggregator
    MultiAggregator
"""
from caldera.gnn.blocks.aggregator import Aggregator
from caldera.gnn.blocks.aggregator import MultiAggregator
from caldera.gnn.blocks.dense import Dense
from caldera.gnn.blocks.edge_block import AggregatingEdgeBlock
from caldera.gnn.blocks.edge_block import EdgeBlock
from caldera.gnn.blocks.flex import Flex
from caldera.gnn.blocks.global_block import AggregatingGlobalBlock
from caldera.gnn.blocks.global_block import GlobalBlock
from caldera.gnn.blocks.node_block import AggregatingNodeBlock
from caldera.gnn.blocks.node_block import NodeBlock

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

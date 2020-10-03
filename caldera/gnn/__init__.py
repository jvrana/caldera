r"""
Graph neural network modules
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
from caldera.gnn.models.encoder_core_decoder import EncodeCoreDecode
from caldera.gnn.models.graph_core import GraphCore
from caldera.gnn.models.graph_encoder import GraphEncoder

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
    "EncodeCoreDecode",
    "GraphCore",
    "GraphEncoder",
]

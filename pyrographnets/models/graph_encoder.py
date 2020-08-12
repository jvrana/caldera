import torch

from pyrographnets.blocks import EdgeBlock
from pyrographnets.blocks import GlobalBlock
from pyrographnets.blocks import NodeBlock
from pyrographnets.data import GraphBatch
from pyrographnets.data import GraphTuple
from pyrographnets.models.base import GraphNetworkBase


class GraphEncoder(GraphNetworkBase):
    def __init__(
        self, edge_block: EdgeBlock, node_block: NodeBlock, global_block: GlobalBlock
    ):
        assert issubclass(type(edge_block), EdgeBlock)
        assert issubclass(type(node_block), NodeBlock)
        assert issubclass(type(global_block), GlobalBlock)
        super().__init__()
        self.node_block = node_block
        self.edge_block = edge_block
        self.global_block = global_block

    def forward(self, data: GraphBatch) -> GraphTuple:
        def run_block(block):
            try:
                return block.forward_from_data(data)
            except RuntimeError as e:
                raise type(e)(
                    "error running `{}.forward()`. {}".format(block._get_name(), str(e))
                ) from e

        edge_attr = run_block(self.edge_block)
        node_attr = run_block(self.node_block)
        global_attr = run_block(self.global_block)
        return GraphTuple(edge_attr, node_attr, global_attr)

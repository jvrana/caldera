from pyrographnets.blocks import EdgeBlock, NodeBlock, GlobalBlock
from pyrographnets.blocks import AggregatingEdgeBlock, AggregatingNodeBlock, AggregatingGlobalBlock
from pyrographnets.data import GraphBatch
import torch
from typing import Tuple


class GraphEncoder(torch.nn.Module):

    def __init__(self, edge_block: EdgeBlock, node_block: NodeBlock, global_block: GlobalBlock):
        assert issubclass(type(edge_block), EdgeBlock)
        assert issubclass(type(node_block), NodeBlock)
        assert issubclass(type(global_block), GlobalBlock)
        super().__init__()
        self.node_block = node_block
        self.edge_block = edge_block
        self.global_block = global_block

    def forward(self, data: GraphBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_attr = self.node_block.forward_from_data(data)
        edge_attr = self.edge_block.forward_from_data(data)
        global_attr = self.global_block.forward_from_data(data)
        return edge_attr, node_attr, global_attr


class GraphCore(torch.nn.Module):

    def __init__(self, edge_block: EdgeBlock, node_block: NodeBlock, global_block: GlobalBlock):
        assert issubclass(type(edge_block), AggregatingEdgeBlock)
        assert issubclass(type(node_block), AggregatingNodeBlock)
        assert issubclass(type(global_block), AggregatingGlobalBlock)
        super().__init__()
        self.node_block = node_block
        self.edge_block = edge_block
        self.global_block = global_block

    def forward(self, data: GraphBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_attr = self.edge_block(data.e, data.x, data.edges)
        node_attr = self.node_block(data.x, edge_attr, data.edges)
        global_attr = self.global_block(data.g, node_attr, edge_attr, data.edges, data.node_idx, data.edge_idx)
        return edge_attr, node_attr, global_attr
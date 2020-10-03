from caldera.data import GraphBatch
from caldera.data import GraphTuple
from caldera.gnn.blocks import AggregatingEdgeBlock
from caldera.gnn.blocks import AggregatingGlobalBlock
from caldera.gnn.blocks import AggregatingNodeBlock
from caldera.gnn.models.base import GraphNetworkBase


class GraphCore(GraphNetworkBase):
    def __init__(
        self,
        *,
        edge_block: AggregatingEdgeBlock,
        node_block: AggregatingNodeBlock,
        global_block: AggregatingGlobalBlock,
        pass_global_to_edge: bool = False,
        pass_global_to_node: bool = False,
    ):
        """A graph network composed of an AggregatingEdgeBlock,
        AggregatingNodeBlock, AggregatingGlobalBlock.

        :param edge_block: AggregatingEdgeBlock to apply
        :param node_block: AggregatingEdgeBlock to apply
        :param global_block: AggregatingEdgeBlock to apply
        :param pass_global_to_edge: Indicates whether to concatenate data from the global block
                                    to the edge block.
        :param pass_global_to_node: Indicates whether to concatenate data from the global block
                                    to the node block.
        """
        assert issubclass(type(edge_block), AggregatingEdgeBlock)
        assert issubclass(type(node_block), AggregatingNodeBlock)
        assert issubclass(type(global_block), AggregatingGlobalBlock)
        super().__init__()
        self.node_block = node_block
        self.edge_block = edge_block
        self.global_block = global_block
        self.pass_to_global_to_edge = pass_global_to_edge
        self.pass_to_global_to_node = pass_global_to_node

    def forward(self, data: GraphBatch) -> GraphTuple:
        if self.pass_to_global_to_edge:
            edge_attr = self.edge_block(
                edge_attr=data.e,
                node_attr=data.x,
                edges=data.edges,
                global_attr=data.g,
                edge_idx=data.edge_idx,
            )
        else:
            edge_attr = self.edge_block(
                edge_attr=data.e, node_attr=data.x, edges=data.edges
            )

        if self.pass_to_global_to_node:
            node_attr = self.node_block(
                node_attr=data.x,
                edge_attr=edge_attr,
                edges=data.edges,
                global_attr=data.g,
                node_idx=data.node_idx,
            )
        else:
            node_attr = self.node_block(
                node_attr=data.x, edge_attr=edge_attr, edges=data.edges
            )

        global_attr = self.global_block(
            global_attr=data.g,
            node_attr=node_attr,
            edge_attr=edge_attr,
            edges=data.edges,
            node_idx=data.node_idx,
            edge_idx=data.edge_idx,
        )
        return GraphTuple(edge_attr, node_attr, global_attr)

from pyrographnets.models import GraphEncoder, GraphCore
from pyrographnets.blocks import EdgeBlock, NodeBlock, GlobalBlock
from pyrographnets.blocks import (
    AggregatingEdgeBlock,
    AggregatingNodeBlock,
    AggregatingGlobalBlock,
)
from pyrographnets.blocks import Aggregator
from pyrographnets.blocks import MLP, Flex
from pyrographnets.data import GraphData, GraphBatch


def test_graph_encoder_forward():
    model = GraphEncoder(
        EdgeBlock(MLP(16, 16)), NodeBlock(MLP(16, 16)), GlobalBlock(MLP(2, 2))
    )

    data = GraphData.random(16, 16, 2)
    batch = GraphBatch.from_data_list([data, data])
    model(batch)


def test_graph_core_init():
    model = GraphCore(
        AggregatingEdgeBlock(Flex(MLP)(Flex.d(), 16)),
        AggregatingNodeBlock(Flex(MLP)(Flex.d(), 16), Aggregator("add")),
        AggregatingGlobalBlock(
            Flex(MLP)(Flex.d(), 2), Aggregator("add"), Aggregator("add")
        ),
    )

    data = random_graph_data(16, 16, 2)
    batch = GraphBatch.from_data_list([data, data])
    model(batch)

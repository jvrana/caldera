from pyrographnets.blocks import MultiAggregator, Flex, MLP, AggregatingEdgeBlock, AggregatingNodeBlock, AggregatingGlobalBlock
from pyrographnets.models import GraphCore

def test_construct_graph_core():

    network = GraphCore(
        edge_block=AggregatingEdgeBlock(
            Flex(MLP)(Flex.d(), 10)
        ),
        node_block=AggregatingNodeBlock(
            Flex(MLP)(Flex.d(), 10),
            edge_aggregator=Flex(MultiAggregator)(Flex.d(), aggregators=['min', 'max', 'add'])
        )
    )
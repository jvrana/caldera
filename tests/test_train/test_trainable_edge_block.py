from pyrographnets.blocks import Flex, MLP, EdgeBlock, AggregatingEdgeBlock
from pyrographnets.data import GraphBatch
import networkx as nx


def test_train_edge_block():

    network = EdgeBlock(Flex(MLP)(Flex.d(), 16, 1))
    batch = GraphBatch.random_batch(100, 5, 5, 5)
    print(batch.e)
    out = network.forward_from_data(batch)
    print(out)

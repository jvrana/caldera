from typing import Tuple, Any, Dict
from pyro_graph_nets.blocks import EdgeBlock, NodeBlock, GlobalBlock, MLP, Aggregator
from pyro_graph_nets.models import GraphEncoder, GraphNetwork
import torch
import numpy as np
from pyro_graph_nets.utils.data import random_input_output_graphs
from pyro_graph_nets.utils.graph_tuple import to_graph_tuple
from pyro_graph_nets.flex import Flex, FlexBlock, FlexDim


class TestFlexBlock(object):

    def test_flex_block(self):
        flex_linear = Flex(torch.nn.Linear)
        model = flex_linear(Flex.d(), 11)
        print(model)
        x = torch.randn((30, 55))
        model(x)
        print(model)

def graph_generator(
        n_nodes: Tuple[int, int],
        n_features: Tuple[int, int],
        e_features: Tuple[int, int],
        g_features: Tuple[int, int]
):
    gen = random_input_output_graphs(
        lambda: np.random.randint(*n_nodes),
        20,
        lambda: np.random.uniform(1, 10, n_features[0]),
        lambda: np.random.uniform(1, 10, e_features[0]),
        lambda: np.random.uniform(1, 10, g_features[0]),
        lambda: np.random.uniform(1, 10, n_features[1]),
        lambda: np.random.uniform(1, 10, e_features[1]),
        lambda: np.random.uniform(1, 10, g_features[1]),
        input_attr_name='features',
        target_attr_name='target',
        do_copy=False
    )
    return gen


class MetaTest(object):

    def generator(self, n_nodes=(2, 20)):
        return graph_generator(
            n_nodes, (10, 1), (5, 2), (1, 3)
        )

    def input_target(self, n_nodes=(2, 20)):
        gen = self.generator(n_nodes)
        graphs = [next(gen) for _ in range(100)]
        assert graphs
        input_gt = to_graph_tuple(graphs)
        target_gt = to_graph_tuple(graphs, feature_key='target')
        return input_gt, target_gt


class TestFlexibleModel(MetaTest):


    def test_flex_encoder(self):
        input_gt, target_gt = self.input_target()
        encoder = GraphEncoder(
            EdgeBlock(FlexBlock(MLP, FlexDim(), 16, 16), independent=True),
            NodeBlock(FlexBlock(MLP, FlexDim(), 16, 16), independent=True),
            None
        )
        print(encoder)

        encoder(input_gt)


    def test_flex_network_0(self):
        input_gt, target_gt = self.input_target()
        FlexMLP = Flex(MLP)
        network = GraphNetwork(
            EdgeBlock(FlexMLP(Flex.d(), 16, 16), independent=False),
            NodeBlock(FlexMLP(Flex.d(), 16, 16), independent=False, edge_aggregator=Aggregator('mean')),
            None
        )
        print(network)
        network(input_gt)
        print(network)


    def test_flex_network_0(self):
        input_gt, target_gt = self.input_target()
        FlexMLP = Flex(MLP)
        network = GraphNetwork(
            EdgeBlock(FlexMLP(Flex.d(), 16, 16), independent=False),
            NodeBlock(FlexMLP(Flex.d(), 16, 16), independent=False, edge_aggregator=Aggregator('mean')),
            GlobalBlock(FlexMLP(Flex.d(), 16, 2), independent=False, edge_aggregator=Aggregator('add'),
                        node_aggregator=Aggregator('mean'))
        )
        print(network)
        network(input_gt)
        print(network)
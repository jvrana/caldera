from typing import Tuple, Any, Dict
from pyro_graph_nets.blocks import EdgeBlock, NodeBlock, GlobalBlock, MLP, Aggregator
from pyro_graph_nets.models import GraphEncoder, GraphNetwork
import torch
import numpy as np
from pyro_graph_nets.utils.data import random_input_output_graphs, GraphDataLoader, GraphDataset
from pyro_graph_nets.utils.graph_tuple import to_graph_tuple, cat_gt
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


class EncodeProcessDecode(torch.nn.Module):

    def __init__(self):
        super().__init__()
        FlexMLP = Flex(MLP)
        self.encoder = GraphEncoder(
            EdgeBlock(FlexMLP(Flex.d(), 16, 16), independent=True),
            NodeBlock(FlexMLP(Flex.d(), 16, 16), independent=True),
            GlobalBlock(FlexMLP(Flex.d(), 16, 16), independent=True)
        )

        # note that core should have the same output dimensions as the encoder
        self.core = GraphNetwork(
            EdgeBlock(FlexMLP(Flex.d(), 16, 16),
                      independent=False),
            NodeBlock(FlexMLP(Flex.d(), 16, 16),
                      independent=False,
                      edge_aggregator=Aggregator('mean')),
            GlobalBlock(FlexMLP(Flex.d(), 16, 16),
                        independent=False,
                        edge_aggregator=Aggregator('mean'),
                        node_aggregator=Aggregator('mean'))
        )

        self.decoder = GraphEncoder(
            EdgeBlock(FlexMLP(Flex.d(), 16, 1), independent=True),
            NodeBlock(FlexMLP(Flex.d(), 16, 1), independent=True),
            GlobalBlock(FlexMLP(Flex.d(), 16, 1), independent=True)
        )

        self.output_transform = GraphEncoder(
            EdgeBlock(Flex(torch.nn.Linear)(Flex.d(), 1), independent=True),
            NodeBlock(Flex(torch.nn.Linear)(Flex.d(), 1), independent=True),
            GlobalBlock(Flex(torch.nn.Linear)(Flex.d(), 1), independent=True),
        )

    def forward(self, input_gt, num_steps: int):
        latent = self.encoder(input_gt)
        latent0 = latent

        output = []
        for step in range(num_steps):
            core_input = cat_gt(latent0, latent)
            latent = self.core(core_input)
            decoded = self.output_transform(self.decoder(latent))
            output.append(decoded)
        return output


class TestFlexEncodeProcessDecode(MetaTest):

    def test_forward(self):
        input_gt, target_gt = self.input_target()
        model = EncodeProcessDecode()
        model(input_gt, 10)

    def test_training(self):
        generator = graph_generator(
            (2, 25), (10, 1), (5, 2), (1, 3)
        )

        graphs = [next(generator) for _ in range(1000)]

        dataset = GraphDataset(graphs)
        loader = GraphDataLoader(dataset, batch_size=50, shuffle=True)
        model = EncodeProcessDecode()

        optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
        criterion = torch.nn.BCEWithLogitsLoss()

        running_loss = []
        num_epochs = 10
        for epoch in range(num_epochs):
            # min batch
            for batch_ndx, bg in enumerate(loader):
                input_gt = to_graph_tuple(bg, feature_key='features')
                target_gt = to_graph_tuple(bg, feature_key='target')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(input_gt)
                loss = criterion(outputs, target_gt)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
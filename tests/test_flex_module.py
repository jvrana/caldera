from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import pytest
import torch
from flaky import flaky

from pyro_graph_nets.blocks import Aggregator
from pyro_graph_nets.blocks import EdgeBlock
from pyro_graph_nets.blocks import GlobalBlock
from pyro_graph_nets.blocks import MLP
from pyro_graph_nets.blocks import NodeBlock
from pyro_graph_nets.flex import Flex
from pyro_graph_nets.flex import FlexBlock
from pyro_graph_nets.flex import FlexDim
from pyro_graph_nets.models import GraphEncoder
from pyro_graph_nets.models import GraphNetwork
from pyro_graph_nets.utils.data import GraphDataLoader
from pyro_graph_nets.utils.data import GraphDataset
from pyro_graph_nets.utils.data import random_input_output_graphs
from pyro_graph_nets.utils.graph_tuple import cat_gt
from pyro_graph_nets.utils.graph_tuple import print_graph_tuple_shape
from pyro_graph_nets.utils.graph_tuple import to_graph_tuple
from pyro_graph_nets.utils.graph_tuple import validate_gt


class TestFlexBlock:
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
    g_features: Tuple[int, int],
):
    a, b = 1, 3
    gen = random_input_output_graphs(
        lambda: np.random.randint(*n_nodes),
        20,
        lambda: np.random.uniform(a, b, n_features[0]),
        lambda: np.random.uniform(a, b, e_features[0]),
        lambda: np.random.uniform(a, b, g_features[0]),
        lambda: np.random.uniform(a, b, n_features[1]),
        lambda: np.random.uniform(a, b, e_features[1]),
        lambda: np.random.uniform(a, b, g_features[1]),
        input_attr_name="features",
        target_attr_name="target",
        do_copy=False,
    )
    return gen


@pytest.mark.parametrize("n_graphs", [1, 10, 100, 500])
@pytest.mark.parametrize("key", ["features", "target"])
def test_generator(n_graphs, key):
    """GraphTuples should always be valid."""
    gen = graph_generator((30, 50), (10, 1), (9, 1), (8, 1))
    graphs = []
    for _ in range(n_graphs):
        graphs.append(next(gen))

    input_gt = to_graph_tuple(graphs, feature_key=key)
    validate_gt(input_gt)


def test_validate_data_loader():
    n_graphs = 1000
    gen = graph_generator((30, 50), (10, 1), (9, 1), (8, 1))
    graphs = []
    for _ in range(n_graphs):
        graphs.append(next(gen))

    dataset = GraphDataset(graphs)
    loader = GraphDataLoader(dataset, shuffle=True, batch_size=100)
    for g in loader:
        input_gt = to_graph_tuple(g)
        target_gt = to_graph_tuple(g, feature_key="target")
        validate_gt(input_gt)
        validate_gt(target_gt)


class MetaTest:
    def generator(self, n_nodes=(2, 20)):
        return graph_generator(n_nodes, (10, 1), (5, 2), (1, 3))

    def input_target(self, n_nodes=(2, 20)):
        gen = self.generator(n_nodes)
        graphs = [next(gen) for _ in range(100)]
        assert graphs
        input_gt = to_graph_tuple(graphs)
        target_gt = to_graph_tuple(graphs, feature_key="target")
        return input_gt, target_gt


class TestFlexibleModel(MetaTest):
    def test_flex_encoder(self):
        input_gt, target_gt = self.input_target()
        encoder = GraphEncoder(
            EdgeBlock(FlexBlock(MLP, FlexDim(), 16, 16), independent=True),
            NodeBlock(FlexBlock(MLP, FlexDim(), 16, 16), independent=True),
            None,
        )
        print(encoder)

        out = encoder(input_gt)

        assert out.node_attr.requires_grad
        assert out.edge_attr.requires_grad
        assert out.global_attr.requires_grad

    def test_flex_network_0(self):
        input_gt, target_gt = self.input_target()
        FlexMLP = Flex(MLP)
        network = GraphNetwork(
            EdgeBlock(FlexMLP(Flex.d(), 16, 16), independent=False),
            NodeBlock(
                FlexMLP(Flex.d(), 16, 16),
                independent=False,
                edge_aggregator=Aggregator("mean"),
            ),
            None,
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
            GlobalBlock(FlexMLP(Flex.d(), 16, 16), independent=True),
        )

        # note that core should have the same output dimensions as the encoder
        self.core = GraphNetwork(
            EdgeBlock(FlexMLP(Flex.d(), 16, 16), independent=False),
            NodeBlock(
                FlexMLP(Flex.d(), 16, 16),
                independent=False,
                edge_aggregator=Aggregator("mean"),
            ),
            GlobalBlock(
                FlexMLP(Flex.d(), 16, 16),
                independent=False,
                edge_aggregator=Aggregator("mean"),
                node_aggregator=Aggregator("mean"),
            ),
        )

        self.decoder = GraphEncoder(
            EdgeBlock(FlexMLP(Flex.d(), 16, 2), independent=True),
            NodeBlock(FlexMLP(Flex.d(), 16, 1), independent=True),
            GlobalBlock(FlexMLP(Flex.d(), 16, 3), independent=True),
        )

        self.output_transform = GraphEncoder(
            EdgeBlock(Flex(torch.nn.Linear)(Flex.d(), 2), independent=True),
            NodeBlock(Flex(torch.nn.Linear)(Flex.d(), 1), independent=True),
            GlobalBlock(Flex(torch.nn.Linear)(Flex.d(), 3), independent=True),
        )

    def forward(self, input_gt, num_steps: int):
        latent = self.encoder(input_gt)
        latent0 = latent

        output = []
        for step in range(num_steps):
            core_input = cat_gt(latent0, latent)
            latent = self.core(core_input)
            decoded = self.decoder(latent)
            out = self.output_transform(decoded)
            output.append(decoded)
        return output


class TestFlexEncodeProcessDecode(MetaTest):
    @flaky(max_runs=10, min_passes=10)
    def test_forward(self):
        input_gt, target_gt = self.input_target()
        model = EncodeProcessDecode()
        out = model(input_gt, 10)
        out = model(input_gt, 10)

    def test_forward_with_data_loader(self):
        generator = graph_generator((2, 25), (10, 1), (5, 2), (1, 3))

        graphs = [next(generator) for _ in range(1000)]

        dataset = GraphDataset(graphs)
        model = EncodeProcessDecode()

        # prime the model
        input_gt = to_graph_tuple([dataset[0]], feature_key="features")

        validate_gt(input_gt)
        outputs = model(input_gt, 10)

        print(outputs[0].node_attr[0, :10])
        print(outputs[-1].node_attr[0, :10])

    def test_loss(self):
        input_gt, target_gt = self.input_target()
        model = EncodeProcessDecode()
        outputs = model(input_gt, 10)
        print_graph_tuple_shape(outputs[-1])
        print_graph_tuple_shape(target_gt)

        criterion = torch.nn.MSELoss()
        loss = 0.0
        loss += criterion(outputs[-1].node_attr, target_gt.node_attr)
        loss += criterion(outputs[-1].edge_attr, target_gt.edge_attr)
        loss += criterion(outputs[-1].global_attr, target_gt.global_attr)
        print(loss)

    # TODO: demonstrate cuda training
    def test_training(self, new_writer):
        writer = new_writer("test_encoder_decoder", suffix="_test")

        generator = graph_generator((2, 25), (10, 1), (5, 2), (1, 3))

        graphs = [next(generator) for _ in range(100)]

        # training loader
        dataset = GraphDataset(graphs)
        n_train = int((len(dataset) * 0.9))
        n_test = len(dataset) - n_train
        train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test])
        loader = GraphDataLoader(train_set, batch_size=50, shuffle=True)
        test_loader = GraphDataLoader(test_set, batch_size=50, shuffle=False)

        model = EncodeProcessDecode()

        # prime the model
        input_gt = to_graph_tuple([dataset[0]], feature_key="features")
        with torch.no_grad():
            model(input_gt, 10)

        optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
        criterion = torch.nn.MSELoss()

        def loss_fn(outputs, target_gt):
            loss = 0.0
            loss += criterion(outputs[-1].node_attr, target_gt.node_attr)
            loss += criterion(outputs[-1].edge_attr, target_gt.edge_attr)
            loss += criterion(outputs[-1].global_attr, target_gt.global_attr)
            return loss

        running_loss = 0.0
        num_epochs = 50
        num_steps = 10
        for epoch in range(num_epochs):

            # min batch
            for batch_ndx, bg in enumerate(loader):
                input_gt = to_graph_tuple(bg, feature_key="features")
                target_gt = to_graph_tuple(bg, feature_key="target")

                validate_gt(input_gt)
                validate_gt(target_gt)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(input_gt, num_steps)

                for out in outputs:
                    validate_gt(out)
                loss = loss_fn(outputs, target_gt)
                loss.backward()
                optimizer.step()
                #
                running_loss += loss.item()

            with torch.no_grad():
                running_test_loss = 0.0
                for test_batch in test_loader:
                    test_input_gt = to_graph_tuple(test_batch, feature_key="features")
                    test_target_gt = to_graph_tuple(test_batch, feature_key="target")
                    test_outputs = model(test_input_gt, num_steps)
                    test_loss = loss_fn(test_outputs, test_target_gt)
                    running_test_loss += test_loss.item()
            writer.add_scalar("test_loss", running_test_loss / 1000.0, epoch)

            writer.add_scalar("training loss", running_loss / 1000, epoch)
            running_loss = 0.0

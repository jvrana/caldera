import time
from typing import Tuple

import networkx as nx
import numpy as np
import pytest
from flaky import flaky

from pyro_graph_nets.blocks import Aggregator
from pyro_graph_nets.blocks import EdgeBlock
from pyro_graph_nets.blocks import GlobalBlock
from pyro_graph_nets.blocks import MLP
from pyro_graph_nets.blocks import NodeBlock
from pyro_graph_nets.models import GraphEncoder
from pyro_graph_nets.models import GraphNetwork
from pyro_graph_nets.utils.data import add_features
from pyro_graph_nets.utils.data import GraphDataLoader
from pyro_graph_nets.utils.data import GraphDataset
from pyro_graph_nets.utils.data import random_graph_generator
from pyro_graph_nets.utils.data import random_input_output_graphs
from pyro_graph_nets.utils.graph_tuple import GraphTuple
from pyro_graph_nets.utils.graph_tuple import to_graph_tuple


def graph_generator(
    n_nodes: Tuple[int, int],
    n_features: Tuple[int, int],
    e_features: Tuple[int, int],
    g_features: Tuple[int, int],
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
        input_attr_name="features",
        target_attr_name="target",
        do_copy=False,
    )
    return gen


class TestGraphGenerator:
    def generator(self):
        return graph_generator((1, 20), (10, 1), (5, 2), (1, 3))

    def get_example(self):
        return next(self.generator())

    def test_check_digraph(self):
        graph = self.get_example()
        assert isinstance(graph, nx.DiGraph)

    def test_check_node_features(self):
        graph = self.get_example()
        for _, ndata in graph.nodes(data=True):
            assert "features" in ndata
            assert ndata["features"].shape[0] == 10

    def test_check_node_target(self):
        graph = self.get_example()
        for _, ndata in graph.nodes(data=True):
            assert "target" in ndata
            assert ndata["target"].shape[0] == 1

    def test_check_edge_features(self):
        graph = self.get_example()
        for _, _, edata in graph.edges(data=True):
            assert "features" in edata
            assert edata["features"].shape[0] == 5

    def test_check_edge_target(self):
        graph = self.get_example()
        for _, _, edata in graph.edges(data=True):
            assert "target" in edata
            assert edata["target"].shape[0] == 2

    def test_global_feature(self):
        # check global data
        graph = self.get_example()
        assert hasattr(graph, "data")
        assert "features" in graph.data
        assert graph.data["features"].shape[0] == 1

    def test_global_target(self):
        # check global data
        graph = self.get_example()
        assert hasattr(graph, "data")
        assert "target" in graph.data
        assert graph.data["target"].shape[0] == 3


class TestGraphDataLoader:
    def generator(self):
        return graph_generator((2, 20), (10, 1), (5, 2), (1, 3))

    def test_dataset(self):
        gen = self.generator()
        graphs = [next(gen) for _ in range(100)]
        dataset = GraphDataset(graphs)
        assert isinstance(dataset[0], nx.DiGraph)

    def test_dataloader(self):
        gen = self.generator()
        graphs = [next(gen) for _ in range(100)]
        dataset = GraphDataset(graphs)
        dataloader = GraphDataLoader(dataset, shuffle=True, batch_size=10)

        for batch_ndx, batched_graphs in enumerate(dataloader):
            input_gt = to_graph_tuple(batched_graphs)
            target_gt = to_graph_tuple(batched_graphs, feature_key="target")

            assert isinstance(input_gt, GraphTuple)
            assert input_gt.node_attr.shape[1] == 10
            assert input_gt.edge_attr.shape[1] == 5
            assert input_gt.global_attr.shape[1] == 1

            assert isinstance(target_gt, GraphTuple)
            assert target_gt.node_attr.shape[1] == 1
            assert target_gt.edge_attr.shape[1] == 2
            assert target_gt.global_attr.shape[1] == 3

        assert batched_graphs


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


class TestEncoder(MetaTest):
    def test_node_block(self):

        model = GraphEncoder(None, NodeBlock(MLP(10, 16, 5), independent=True), None)
        input_gt, target_gt = self.input_target()
        out = model(input_gt)
        assert out

    def test_edge_block(self):

        model = GraphEncoder(EdgeBlock(MLP(5, 16, 5), independent=True), None, None)
        input_gt, target_gt = self.input_target()
        out = model(input_gt)
        assert out

    def test_global_block(self):

        model = GraphEncoder(None, None, GlobalBlock(MLP(1, 16, 2), independent=True))
        input_gt, target_gt = self.input_target()
        out = model(input_gt)
        assert out

    def test_empty_graph_global_block(self):
        model = GraphEncoder(None, None, GlobalBlock(MLP(1, 16, 2), independent=True))
        input_gt, target_gt = self.input_target((1, 2))
        out = model(input_gt)
        assert out

    def test_all(self):

        model = GraphEncoder(
            EdgeBlock(MLP(5, 16, 15), independent=True),
            NodeBlock(MLP(10, 16, 5), independent=True),
            GlobalBlock(MLP(1, 16, 2), independent=True),
        )
        input_gt, target_gt = self.input_target()

        out = model(input_gt)

        assert out.node_attr.requires_grad
        assert out.edge_attr.requires_grad
        assert out.global_attr.requires_grad

        assert out.edge_attr.shape[1] == 15
        assert out.node_attr.shape[1] == 5
        assert out.global_attr.shape[1] == 2


class TestNetwork(MetaTest):
    def test_node_block(self):

        model = GraphNetwork(
            None,
            NodeBlock(
                MLP(10 + 5, 16, 5),
                independent=False,
                edge_aggregator=Aggregator("mean"),
            ),
            None,
        )
        input_gt, target_gt = self.input_target()
        out = model(input_gt)
        assert out

    def test_edge_block(self):

        model = GraphNetwork(
            EdgeBlock(MLP(10 + 10 + 5, 16, 5), independent=False), None, None
        )
        input_gt, target_gt = self.input_target()
        out = model(input_gt)
        assert out

    def test_global_block(self):

        model = GraphNetwork(
            None,
            None,
            GlobalBlock(
                MLP(10 + 5 + 1, 16, 2),
                independent=False,
                node_aggregator=Aggregator("mean"),
                edge_aggregator=Aggregator("mean"),
            ),
        )
        input_gt, target_gt = self.input_target()
        out = model(input_gt)
        assert out

    def test_all(self):

        model = GraphNetwork(
            EdgeBlock(MLP(25, 16, 15), independent=False),
            NodeBlock(
                MLP(25, 16, 5), independent=False, edge_aggregator=Aggregator("mean")
            ),
            GlobalBlock(
                MLP(21, 16, 2),
                independent=False,
                node_aggregator=Aggregator("mean"),
                edge_aggregator=Aggregator("mean"),
            ),
        )
        input_gt, target_gt = self.input_target()

        out = model(input_gt)
        assert out.edge_attr.shape[1] == 15
        assert out.node_attr.shape[1] == 5
        assert out.global_attr.shape[1] == 2

    @pytest.mark.parametrize(
        "agg", ["mean", "max", "min", "add"], ids=["mean", "max", "min", "add"]
    )
    @pytest.mark.parametrize(
        "which_agg",
        [0, 1, 2],
        ids=["node_block_edge", "global_block_node", "global_block_edge"],
    )
    @flaky(max_runs=5, min_passes=5)
    def test_all_aggregators(self, agg, which_agg):
        aggs = ["mean", "mean", "mean"]
        aggs[which_agg] = agg
        model = GraphNetwork(
            EdgeBlock(MLP(25, 16, 15), independent=False),
            NodeBlock(
                MLP(25, 16, 5), independent=False, edge_aggregator=Aggregator(aggs[0])
            ),
            GlobalBlock(
                MLP(21, 16, 2),
                independent=False,
                node_aggregator=Aggregator(aggs[1]),
                edge_aggregator=Aggregator(aggs[2]),
            ),
        )
        input_gt, target_gt = self.input_target()

        out = model(input_gt)
        assert out.edge_attr.shape[1] == 15
        assert out.node_attr.shape[1] == 5
        assert out.global_attr.shape[1] == 2

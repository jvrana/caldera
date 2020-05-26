from typing import Tuple, Any, Dict
from pyro_graph_nets.blocks import EdgeBlock, NodeBlock
from pyro_graph_nets.models import GraphEncoder, GraphNetwork
import torch
import numpy as np
from pyro_graph_nets.utils.data import random_input_output_graphs
from pyro_graph_nets.utils.graph_tuple import to_graph_tuple


class FlexDim(object):

    def __init__(self, arg_pos: int = 0):
        self.arg_pos = arg_pos

    def resolve(self, input_args, input_kwargs):
        return input_args[self.arg_pos].shape[1]


class UnsetModule(torch.nn.Module):

    def __init__(self, module_fn, *args, **kwargs):
        super().__init__()
        self.module = module_fn
        self.args = args
        self.kwargs = kwargs
        self.resolved_module = None

    def resolve_args(self, input_args: Tuple[Any, ...], input_kwargs: Dict[str, Any]):
        rargs = []
        for i, a in enumerate(self.args):
            if isinstance(a, FlexDim):
                rargs.append(a.resolve(input_args, input_kwargs))
            else:
                rargs.append(a)
        return rargs

    def resolve_kwargs(self, input_args: Tuple[Any, ...], input_kwargs: Dict[str, Any]):
        return self.kwargs

    def resolve(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        resolved_args = self.resolve_args(args, kwargs)
        resolved_kwargs = self.resolve_kwargs(args, kwargs)
        self.resolved_module = self.module(*resolved_args, **resolved_kwargs)

    def forward(self, *args, **kwargs):
        if self.resolved_module is None:
            self.resolve(args, kwargs)
        return self.resolved_module(*args, **kwargs)


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
            UnsetModule(EdgeBlock, FlexDim(2), [16, 16], independent=True),
            UnsetModule(NodeBlock, FlexDim(0), [16, 16], independent=True),
            None
        )
        print(encoder)

        encoder(input_gt)


    def test_flex_network(self):
        input_gt, target_gt = self.input_target()
        encoder = GraphNetwork(
            UnsetModule(EdgeBlock, FlexDim(2), [16, 16], independent=False),
            UnsetModule(NodeBlock, FlexDim(0), [16, 16], independent=False),
            None
        )
        print(encoder)

        encoder(input_gt)
"""
test_train_networks.py

Inststructions for creating a new test case.

loader, getter, network
"""

from typing import Union, Callable, Tuple, Any, Dict, Optional, Type

import pytest
import torch
from torch import optim

from pyrographnets.blocks import NodeBlock, EdgeBlock, GlobalBlock, Flex, MLP
from pyrographnets.models import GraphEncoder
from pyrographnets.data import GraphData, GraphBatch, GraphDataLoader
from pyrographnets.utils import deterministic_seed
import networkx as nx
from pyrographnets.utils.torch_utils import to_one_hot
import numpy as np
import functools


SEED = 0


def name_module(n, m):
    m.name = n
    return m


class Networks(object):
    """Networks that will be used in the tests"""

    n = name_module

    linear_block = n(
        'linear',
        torch.nn.Sequential(
            torch.nn.Linear(5, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
    )

    mlp_block = n(
        'mlp',
        torch.nn.Sequential(
            Flex(MLP)(Flex.d(), 16),
            Flex(torch.nn.Linear)(Flex.d(), 1)
        )
    )

    node_block = n(
        'node_block',
        torch.nn.Sequential(
            NodeBlock(Flex(MLP)(Flex.d(), 25, 25, layer_norm=False)),
            Flex(torch.nn.Linear)(Flex.d(), 1)
        ))

    edge_block = n(
        'edge_block',
        torch.nn.Sequential(
            EdgeBlock(Flex(MLP)(Flex.d(), 25, 25, layer_norm=False)),
            Flex(torch.nn.Linear)(Flex.d(), 1)
        ))

    global_block = n(
        'global_block',
        torch.nn.Sequential(
            GlobalBlock(Flex(MLP)(Flex.d(), 25, 25, layer_norm=False)),
            Flex(torch.nn.Linear)(Flex.d(), 1)
        ))
    graph_encoder = n(
        'graph_encoder',
        GraphEncoder(
            EdgeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), 5, 5, layer_norm=False),
                    Flex(torch.nn.Linear)(Flex.d(), 1)
                )
            ),
            NodeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), 5, 5, layer_norm=False),
                    Flex(torch.nn.Linear)(Flex.d(), 1)
                )
            ),
            GlobalBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), 5, 5, layer_norm=False),
                    Flex(torch.nn.Linear)(Flex.d(), 1)
                )
            ),
        )
    )

    @staticmethod
    def reset(net: torch.nn.Module):
        def weight_reset(model):
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        net.apply(weight_reset)


class DataModifier(object):
    """Methods to modify data before training"""

    def __init__(self, datalist):
        self.datalist = datalist

    @staticmethod
    def node_sum(batch: GraphBatch, copy=True):
        if copy:
            batch = batch.copy()
        batch.x = torch.cat([
            batch.x,
            batch.x.sum(axis=1, keepdim=True)
        ], axis=1)
        return batch

    @staticmethod
    def edge_sum(batch: GraphBatch, copy=True):
        if copy:
            batch = batch.copy()
        batch.e = torch.cat([
            batch.e,
            batch.e.sum(axis=1, keepdim=True)
        ], axis=1)
        return batch

    @staticmethod
    def global_sum(batch: GraphBatch, copy=True):
        if copy:
            batch = batch.copy()
        batch.g = torch.cat([
            batch.g,
            batch.g.sum(axis=1, keepdim=True)
        ], axis=1)
        return batch

    def apply(self, f, *args, **kwargs):
        f = self.resolve(f)
        return [f(_d, *args, **kwargs) for _d in self.datalist]

    @classmethod
    def resolve(cls, f):
        cls.valid(f)
        if isinstance(f, str):
            f = getattr(cls, f)
        return f

    @classmethod
    def valid(self, f):
        if callable(f):
            return True
        elif isinstance(f, str) and hasattr(self, f):
            return True
        return False


class DataLoaders(object):

    @staticmethod
    def random_loader(data_size, batch_size):
        datalist = [GraphData.random(5, 5, 5) for _ in
                    range(data_size)]
        return GraphDataLoader(datalist, batch_size)

    @staticmethod
    def _default_g(g: nx.DiGraph):
        for _, data in g.nodes(data=True):
            data['features'] = np.zeros((1,))
            data['target'] = np.zeros((1,))

        for _, _, data in g.edges(data=True):
            data['features'] = np.zeros((1,))
            data['target'] = np.zeros((1,))

        g.data = {'features': np.zeros((1,)), 'target': np.zeros((1,))}
        return g

    @classmethod
    def random_graph_red_black_nodes(cls, data_size, batch_size):
        input_data = []
        output_data = []
        s = 2
        for _ in range(data_size):
            g = nx.to_directed(nx.random_tree(10))
            cls._default_g(g)
            for n, ndata in g.nodes(data=True):
                i = np.random.randint(0, 1, (1,))
                ndata['features'] = to_one_hot(i, s)
                if i % 2 == 0:
                    target = np.array([0.5])
                else:
                    target = np.zeros(1)
                ndata['target'] = target

            input_data.append(GraphData.from_networkx(g, feature_key='features'))
            output_data.append(GraphData.from_networkx(g, feature_key='target'))

        return GraphDataLoader(list(zip(input_data, output_data)), batch_size=batch_size)

    @classmethod
    def random_graph_red_black_edges(cls, data_size, batch_size):
        input_data = []
        output_data = []
        s = 2
        for _ in range(data_size):
            g = nx.to_directed(nx.random_tree(10))
            cls._default_g(g)
            for _, _, edata in g.edges(data=True):
                i = np.random.randint(0, 1, (1,))
                edata['features'] = to_one_hot(i, s)
                if i % 2 == 0:
                    target = np.array([0.5])
                else:
                    target = np.zeros(1)
                edata['target'] = target

            input_data.append(GraphData.from_networkx(g, feature_key='features'))
            output_data.append(GraphData.from_networkx(g, feature_key='target'))

        return GraphDataLoader(list(zip(input_data, output_data)), batch_size=batch_size)


data_loaders = {
    'random': DataLoaders.random_loader
}


T = Tuple[Tuple[Tuple[Any, ...], Dict], torch.Tensor]


class DataGetter(object):
    """Methods to collect input, output from the loader"""

    @classmethod
    def get_node(cls, batch: GraphBatch) -> T:
        args = (
            batch.x[:, :-1],
        )
        kwargs = {}
        out = batch.x[:, -1:]
        return ((args, kwargs), out)

    @classmethod
    def get_edge(cls, batch: GraphBatch) -> T:
        args = (
            batch.e[:, :-1],
        )
        kwargs = {}
        out = batch.e[:, -1:]
        return ((args, kwargs), out)

    @classmethod
    def get_global(cls, batch: GraphBatch) -> T:
        args = (
            batch.g[:, :-1],
        )
        kwargs = {}
        out = batch.g[:, -1:]
        return ((args, kwargs), out)

    @classmethod
    def get_batch(cls, batch_tuple: Tuple[GraphBatch, GraphBatch]) -> T:
        args = (
            batch_tuple[0],
        )
        kwargs = {}
        out = batch_tuple[1]
        return ((args, kwargs), (out.e, out.x, out.g))



class NetworkTestCaseValidationError(Exception):
    pass

# TODO: model reset is not working
class NetworkTestCase(object):
    """A network test case."""

    def __init__(self,
                 network: torch.nn.Module,
                 modifier: Optional[Callable[[GraphBatch], Any]] = None,
                 getter: Optional[Callable[[GraphBatch], Any]] = None,
                 optimizer: Type[torch.optim.Optimizer] = None,
                 criterion=None,
                 loss_func: Callable = None,
                 epochs: int = 20,
                 batch_size: int = 100,
                 data_size: int = 1000,
                 loader: Optional[Callable[[int, int], GraphDataLoader]] = None):
        if modifier is None:
            self.modifier = lambda x: x
        else:
            self.modifier = modifier
        if getter is None:
            self.getter = lambda x: x
        else:
            self.getter = getter
        self.network = network
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_size = data_size
        self.device = None
        if loader is None:
            loader = DataLoaders.random_loader
        self.loader = loader(data_size, batch_size)
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.losses = None

    def to(self, x, device=None):
        device = device or self.device
        if device is not None:
            if isinstance(x, tuple):
                return tuple([self.to(_x) for _x in x])
            else:
                return x.to(device)
        return x

    def seed(self, seed: int = SEED):
        deterministic_seed(seed)

    def reset(self, seed: int = SEED):
        self.seed(seed)
        Networks.reset(self.network)
        self.to(self.network)

    def provide_example(self):
        batch = self.loader.first()
        mod_batch = self.modifier(batch)
        mod_batch = self.to(mod_batch)
        data = self.getter(mod_batch)[0]
        self.network(*data[0], **data[1])

    # def validate_network_device(self):
    #     for p in self.network.parameters():
    #         assert p.device == self.device

    def train(self):
        print("Training {}".format(self.network))
        self.reset()
        epochs = self.epochs
        net = self.network
        device = self.device
        loader = self.loader
        criterion = self.criterion
        optimizer = self.optimizer
        getter = self.getter
        modifier = self.modifier
        loss_func = self.loss_func

        # provide example
        self.provide_example()

        if optimizer is None:
            optimizer = optim.AdamW(net.parameters(), lr=1e-2)
        if criterion is None:
            criterion = torch.nn.MSELoss()
        if loss_func is not None:
            loss_func = functools.partial(loss_func, criterion, device)
        else:
            loss_func = criterion

        self.pre_train_validate()

        loss_arr = torch.zeros(epochs)
        for epoch in range(epochs):
            net.train()
            running_loss = 0.
            for batch in loader:

                batch = self.to(batch)
                batch = modifier(batch)
                input, target = getter(batch)

                optimizer.zero_grad()  # zero the gradient buffers
                output = net(*input[0], **input[1])

                for x, o, t in zip(['edge', 'node', 'global'], output, target):
                    if o.shape != t.shape:
                        raise NetworkTestCaseValidationError(
                            "{x} output shape ({o}) has a different shape from {x} target shape ({t})".format(
                                x=x, o=o.shape, t=t.shape
                            )
                        )

                loss = loss_func(output, target)
                self.to(loss)
                loss.backward(retain_graph=True)
                optimizer.step()

                running_loss += loss.item()
            loss_arr[epoch] = running_loss
        self.losses = loss_arr
        return loss_arr

    def pre_train_validate(self):
        for p in self.network.parameters():
            assert p.requires_grad is True

    def post_train_validate(self, threshold=0.1):
        if self.losses[-1] > self.losses[0] * threshold:
            raise NetworkTestCaseValidationError("Model did not train properly :(."
                                  "\n\tlosses: {} -> {}".format(
                self.losses[0], self.losses[-1]))

    def __str__(self):
        pass


@pytest.mark.parametrize('loader_func', [
    DataLoaders.random_loader,
    DataLoaders.random_graph_red_black_nodes,
    DataLoaders.random_graph_red_black_edges
])
def test_loaders(loader_func):
    loader = loader_func(100, 20)
    for x in loader:
        assert x


def mse_tuple(criterion, device, a, b):
    loss = torch.tensor(0., dtype=torch.float32, device=device)
    assert len(a) == len(b)
    for i, (_a, _b) in enumerate(zip(a, b)):
        assert _a.shape == _b.shape
        l = criterion(_a, _b)
        loss += l
    return loss


def get_id(case):
    tokens = [case['network'].name]
    loader = case.get('loader', None)
    if loader is not None:
        loader = loader.__name__
        tokens.append(loader)
    print(tokens)
    return '-'.join(tokens)


@pytest.fixture(params=[
    dict(
        network=Networks.linear_block,
        modifier=DataModifier.node_sum,
        getter=DataGetter.get_node
    ),
    dict(
        network=Networks.mlp_block,
        modifier=DataModifier.node_sum,
        getter=DataGetter.get_node
    ),
    dict(
        network=Networks.node_block,
        modifier=DataModifier.node_sum,
        getter=DataGetter.get_node,
    ),
    dict(
        network=Networks.edge_block,
        modifier=DataModifier.edge_sum,
        getter=DataGetter.get_edge,
    ),
    dict(
        network=Networks.global_block,
        modifier=DataModifier.global_sum,
        getter=DataGetter.get_global,
    ),
    dict(
        network=Networks.graph_encoder,
        loader=DataLoaders.random_graph_red_black_nodes,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple
    ),
    dict(
        network=Networks.graph_encoder,
        loader=DataLoaders.random_graph_red_black_edges,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple
    )
], ids=get_id)
def network_case(request):
    return NetworkTestCase(**request.param)


def test_training_cases(network_case, device):
    network_case.device = device
    losses = network_case.train()
    print(losses)
    for p in network_case.network.parameters():
        assert device == str(p.device)
    network_case.post_train_validate()


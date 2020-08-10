from typing import Union, Callable, Tuple, Any, Dict

import pytest
import torch
from torch import optim

from pyrographnets.blocks import NodeBlock, EdgeBlock, GlobalBlock, Flex, MLP
from pyrographnets.data import GraphData, GraphBatch, GraphDataLoader
from pyrographnets.utils import deterministic_seed
from flaky import flaky

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


class NetworkTestCaseValidationError(Exception):
    pass


class NetworkTestCase(object):
    """A network test case."""

    def __init__(self, network, modifier: Union[Callable, str],
                 convert, optimizer=None, criterion=None,
                 epochs: int = 20, batch_size: int = 100, data_size: int = 1000):
        self.modifier = DataModifier.resolve(modifier)
        self.network = network
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_size = data_size
        self.device = None
        self.loader = self.create_loader()
        self.criterion = criterion
        self.optimizer = optimizer
        self.getter = convert
        self.losses = None

    def to(self, x, device=None):
        device = device or self.device
        if device is not None:
            return x.to(device)
        return x

    def seed(self, seed: int = SEED):
        deterministic_seed(SEED)

    def reset(self, seed: int = SEED):
        # self.seed()
        Networks.reset(self.network)
        self.to(self.network)

    def create_loader(self):
        # self.seed()
        datalist = [GraphData.random(5, 5, 5, requires_grad=True) for _ in
                    range(self.data_size)]
        return GraphDataLoader(datalist, self.batch_size)

    def provide_example(self):
        batch = self.loader.first()
        mod_batch = self.modifier(batch)
        mod_batch = self.to(mod_batch)
        data = self.getter(mod_batch)[0]
        self.to(self.network)
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

        # provide example
        self.provide_example()

        if optimizer is None:
            optimizer = optim.AdamW(net.parameters(), lr=1e-2)
        if criterion is None:
            criterion = torch.nn.MSELoss()

        self.pre_train_validate()

        loss_arr = torch.zeros(epochs)
        for epoch in range(epochs):
            net.train()
            running_loss = 0.
            for batch in loader:

                if device:
                    batch = batch.to(device)
                batch = modifier(batch)
                input, target = getter(batch)

                optimizer.zero_grad()  # zero the gradient buffers
                output = net(*input[0], **input[1])
                loss = criterion(output, target)
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


@pytest.fixture(params=[
    dict(
        network=Networks.linear_block,
        modifier=DataModifier.node_sum,
        convert=DataGetter.get_node
    ),
    dict(
        network=Networks.mlp_block,
        modifier=DataModifier.node_sum,
        convert=DataGetter.get_node
    ),
    dict(
        network=Networks.node_block,
        modifier=DataModifier.node_sum,
        convert=DataGetter.get_node,
    ),
    dict(
        network=Networks.edge_block,
        modifier=DataModifier.edge_sum,
        convert=DataGetter.get_edge,
    ),
    dict(
        network=Networks.global_block,
        modifier=DataModifier.global_sum,
        convert=DataGetter.get_global,
    )
], ids=lambda x: x['network'].name)
def network_case(request):
    return NetworkTestCase(**request.param)

def test_training_cases(network_case, device):
    network_case.device = device
    losses = network_case.train()
    print(losses)
    for p in network_case.network.parameters():
        assert device == str(p.device)
    network_case.post_train_validate()

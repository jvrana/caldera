from caldera._future.flow import Flow
from torch import nn
from caldera import gnn
from caldera.data import GraphBatch
from caldera.testing import check_back_prop
from rich import print
import torch
from pprint import pprint
import pytest
import flaky

MIN_RUNS = 5

def multiple_runs(f):
    return flaky.flaky(MIN_RUNS, min_passes=MIN_RUNS)(f)


@multiple_runs
@pytest.mark.parametrize('layer_norm', [False, True])
def test_check_back_prop_layer_norm(layer_norm):
    """
    Backpropogation should work through the layer normalization layer
    (only if elementwise_affine is false however...)
    """
    class Network(nn.Module):

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(5, 10)
            self.act = nn.ReLU()
            if layer_norm:
                self.layer_norm = nn.LayerNorm(10, elementwise_affine=False)

        def forward(self, x):
            x = self.lin(x)
            x = self.act(x)
            if layer_norm:
                x = self.layer_norm(x)
            return x

    net = Network()
    data = torch.randn(10, 5)
    grads = check_back_prop(net, net(data))
    assert grads['lin.bias'] is True
    assert grads['lin.weight'] is True


@multiple_runs
@pytest.mark.parametrize('layer_norm', [True, False], ids=lambda x: 'layer_norm={}'.format(x))
@pytest.mark.parametrize('dropout', [None, 0.2], ids=lambda x: 'dropout={}'.format(x))
def test_check_back_prop_dense(layer_norm, dropout):
    """
    Backpropogation should work through the layer normalization layer
    (only if elementwise_affine is false however...)
    """

    net = gnn.Dense(5, 5, 10, 10, layer_norm=layer_norm, dropout=dropout)
    data = torch.randn(10, 5)
    grads = check_back_prop(net, net(data))
    pprint(grads)
    assert grads['dense_layers.0.dense.linear.weight'] is True
    assert grads['dense_layers.0.dense.linear.bias'] is True
    assert grads['dense_layers.1.dense.linear.weight'] is True
    assert grads['dense_layers.1.dense.linear.bias'] is True
    assert grads['dense_layers.2.dense.linear.weight'] is True
    assert grads['dense_layers.2.dense.linear.bias'] is True


class TestFlow:

    class MyModule(Flow):

        def __init__(self):
            super().__init__()
            self.node = gnn.Flex(nn.Linear)(..., 1)
            self.edge = gnn.Flex(nn.Linear)(..., 1)
            self.edge_to_node_agg = gnn.Aggregator('add')
            self.register_connection(
                lambda data: data.x,
                self.node
            )
            self.register_connection(
                lambda data: data.e,
                self.edge
            )

            self.register_connection(
                self.edge,
                self.edge_to_node_agg,
                None,
                (lambda data: data.edges[1], lambda data: data.x.shape[0])
            )
            self.register_connection(
                self.edge_to_node_agg,
                self.node
            )

        def forward(self, data):
            return self.propogate(self.node, data)
            # self.propogate(self.node, data)

    def test_init(self):
        """We expect no errors with passed randomized data"""
        net = self.MyModule()
        net.train()
        data = GraphBatch.random_batch(10, 5, 4, 3)
        net(data)

    def test_print(self):
        net = self.MyModule()
        print(net)

    def test_print_parameters(self):
        net = self.MyModule()
        data = GraphBatch.random_batch(10, 5, 4, 3)
        net(data)
        for k, v in net.named_parameters():
            print(k)

    def test_print_modules(self):
        net = self.MyModule()
        data = GraphBatch.random_batch(10, 5, 4, 3)
        net(data)
        for k, v in net.named_modules():
            print(k)

    @multiple_runs
    def test_back_prop_node(self):
        net = self.MyModule()
        data = GraphBatch.random_batch(10, 5, 4, 3)

        # activate all
        assert gnn.Flex.has_unresolved_flex_blocks(net)
        net.propogate(net.edge, data)
        assert gnn.Flex.has_unresolved_flex_blocks(net)
        net.propogate(net.node, data)
        assert not gnn.Flex.has_unresolved_flex_blocks(net)

        grads = check_back_prop(net, net.propogate(net.node, data))
        pprint(grads)
        assert grads['node.resolved_module.weight']
        assert grads['node.resolved_module.bias']
        assert grads['edge.resolved_module.weight']
        assert grads['edge.resolved_module.bias']

    @multiple_runs
    def test_back_prop_node(self):
        net = self.MyModule()
        data = GraphBatch.random_batch(10, 5, 4, 3)

        # activate all
        assert gnn.Flex.has_unresolved_flex_blocks(net)
        net.propogate(net.edge, data)
        assert gnn.Flex.has_unresolved_flex_blocks(net)
        net.propogate(net.node, data)
        assert not gnn.Flex.has_unresolved_flex_blocks(net)

        grads = check_back_prop(net, net.propogate(net.edge, data))
        pprint(grads)
        assert not grads['node.resolved_module.weight']
        assert not grads['node.resolved_module.bias']
        assert grads['edge.resolved_module.weight']
        assert grads['edge.resolved_module.bias']

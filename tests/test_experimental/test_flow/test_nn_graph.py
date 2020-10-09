import inspect
from dataclasses import dataclass
from pprint import pprint

import pytest
import torch
from torch import nn

from caldera import gnn
from caldera._experimental.nngraph import NNGraph
from caldera.testing import check_back_prop

# TODO: clean up these tests
class MyNNGraph(NNGraph):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        pass


def test_basic_callable():
    graph = MyNNGraph()
    lin1 = nn.Linear(10, 1)
    graph.add_edge(lambda x: x, lin1)
    with graph.run():
        graph.propogate(lin1, torch.randn(5, 10))


def test_basic_callable2():
    graph = MyNNGraph()
    lin1 = nn.Linear(3, 1)
    graph.add_edge(lambda x: x[1], lin1)
    data = (torch.randn(5, 10), torch.randn(3, 3))
    with graph.run():
        graph.propogate(lin1, data)


def test_basic_indexing():
    graph = MyNNGraph()

    a, b, c = 4, 10, 12
    lin1 = nn.Linear(a, 4)
    lin2 = nn.Linear(a + b, 1)

    data = (torch.randn(10, a), torch.randn(c, b), torch.randint(0, a, size=(c,)))
    graph.add_edge(lambda data: data[0], lin1)
    graph.add_edge(lambda data: data[1], lin2)
    graph.add_edge(lambda data: data[0], lin2, lambda data: data[2])
    with graph.run():
        graph.propogate(lin2, data)
        graph.propogate(lin1, data)


def test_basic_aggregation():
    class Mod(MyNNGraph):
        def __init__(self):
            super().__init__()
            pass

        def forward(self, data):
            pass

    Mod()(torch.randn(5, 5))
    graph = MyNNGraph()
    lin1 = nn.Linear(4, 1)
    lin2 = nn.Linear(10, 1)
    agg = gnn.Aggregator("add")

    data = (torch.randn(10, 3), torch.randn(12, 10), torch.randint(0, 10, size=(12,)))

    graph.add_edge(lambda data: data[0], lin1)
    graph.add_edge(lambda data: data[1], lin2)
    graph.add_edge(
        lin2,
        lin1,
        aggregation=agg,
        indexer=lambda data: data[2],
        size=lambda data: data[0].shape[0],
    )

    with graph.run():
        graph.propogate(lin1, data)

    print(graph)
    print(inspect.getsource(graph.forward))


@pytest.mark.parametrize("check", ["x", "y", "z"])
def test_forward_complex(check):
    class Foo(NNGraph):
        def __init__(self):
            super().__init__()
            self.x_layer = nn.Sequential(nn.Linear(18, 10), nn.ReLU())
            self.y_layer = nn.Sequential(nn.Linear(18, 10), nn.ReLU())
            self.z_layer = nn.Sequential(nn.Linear(25, 10), nn.ReLU())

            agg = gnn.Aggregator("add")

            self.add_node(self.x_layer, "x_layer")
            self.add_node(self.y_layer, "y_layer")
            self.add_node(self.z_layer, "z_layer")

            self.add_edge(lambda data: data.x, self.x_layer)
            self.add_edge(lambda data: data.y, self.y_layer)
            self.add_edge(lambda data: data.z, self.z_layer)

            self.add_edge(
                lambda data: data.x, self.y_layer, indexer=lambda data: data.y_to_x
            )
            self.add_edge(
                lambda data: data.z, self.y_layer, indexer=lambda data: data.y_to_z
            )
            self.add_edge(
                lambda data: data.z, self.x_layer, indexer=lambda data: data.x_to_z
            )
            self.add_edge(
                self.y_layer,
                self.x_layer,
                aggregation=gnn.Aggregator("add"),
                indexer=lambda data: data.y_to_x,
                size=lambda data: data.x.shape[0],
            )
            self.add_edge(
                self.y_layer,
                self.z_layer,
                aggregation=gnn.Aggregator("add"),
                indexer=lambda data: data.y_to_z,
                size=lambda data: data.z.shape[0],
            )
            self.add_edge(
                self.x_layer,
                self.z_layer,
                aggregation=gnn.Aggregator("add"),
                indexer=lambda data: data.x_to_z,
                size=lambda data: data.z.shape[0],
            )

        def forward(self, data):
            with self.run():
                x = self.propogate(self.x_layer, data)
                y = self.propogate(self.y_layer, data)
                z = self.propogate(self.z_layer, data)
            return x, y, z
            # self.validate_visited()

    foo = Foo()

    @dataclass
    class Data:

        x: torch.Tensor
        y: torch.Tensor
        z: torch.Tensor
        y_to_x: torch.LongTensor
        x_to_z: torch.LongTensor
        y_to_z: torch.LongTensor

    data = Data(
        x=torch.randn(10, 3),
        y=torch.randn(12, 10),
        z=torch.randn(3, 5),
        y_to_x=torch.randint(0, 10, size=(12,)),  # mapping of 1 to 0
        y_to_z=torch.randint(0, 3, size=(12,)),  # mapping of 1 to 2
        x_to_z=torch.randint(0, 3, size=(10,)),  # mapping of 0 to 2
    )

    data2 = Data(
        x=torch.randn(10, 3) + 2,
        y=torch.randn(12, 10) + 2,
        z=torch.randn(3, 5) + 2,
        y_to_x=data.y_to_x,
        y_to_z=data.y_to_z,
        x_to_z=data.x_to_z,
    )

    out1 = foo(data)
    out2 = foo(data2)

    if check == "x":
        x_loss = nn.MSELoss()(out1[0], out2[0])
        x_loss_grads = check_back_prop(model=foo, loss=x_loss)

        expected_x_loss_grads = {
            "nodes.x_layer.0.bias": True,
            "nodes.x_layer.0.weight": True,
            "nodes.y_layer.0.bias": True,
            "nodes.y_layer.0.weight": True,
            "nodes.z_layer.0.bias": False,
            "nodes.z_layer.0.weight": False,
        }
        pprint(x_loss_grads)
        for k, v in expected_x_loss_grads.items():
            print(k)
            assert x_loss_grads[k] is v
    elif check == "y":
        y_loss = nn.MSELoss()(out1[1], out2[1])
        y_loss_grads = check_back_prop(model=foo, loss=y_loss)

        expected_y_loss_grads = {
            "nodes.x_layer.0.bias": False,
            "nodes.x_layer.0.weight": False,
            "nodes.y_layer.0.bias": True,
            "nodes.y_layer.0.weight": True,
            "nodes.z_layer.0.bias": False,
            "nodes.z_layer.0.weight": False,
        }
        pprint(y_loss_grads)
        for k, v in expected_y_loss_grads.items():
            print(k)
            assert y_loss_grads[k] is v
    elif check == "z":
        z_loss = nn.MSELoss()(out1[2], out2[2])
        z_loss_grads = check_back_prop(model=foo, loss=z_loss)

        expected_z_loss_grads = {
            "nodes.x_layer.0.bias": True,
            "nodes.x_layer.0.weight": True,
            "nodes.y_layer.0.bias": True,
            "nodes.y_layer.0.weight": True,
            "nodes.z_layer.0.bias": True,
            "nodes.z_layer.0.weight": True,
        }
        pprint(z_loss_grads)
        for k, v in expected_z_loss_grads.items():
            print(k)
            assert z_loss_grads[k] is v

    # y_loss = nn.MSELoss(out1[0], out2[0])
    # z_loss = nn.MSELoss(out1[0], out2[0])
    #
    # check_back_prop(foo, out=None, loss=x_loss)

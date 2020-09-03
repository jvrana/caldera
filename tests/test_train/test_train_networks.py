"""test_train_networks.py.

Inststructions for creating a new test case.

loader, getter, network
"""
import functools
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type

import networkx as nx
import numpy as np
import pytest
import torch
from torch import optim

from caldera.blocks import AggregatingEdgeBlock
from caldera.blocks import AggregatingGlobalBlock
from caldera.blocks import AggregatingNodeBlock
from caldera.blocks import Aggregator
from caldera.blocks import EdgeBlock
from caldera.blocks import Flex
from caldera.blocks import GlobalBlock
from caldera.blocks import MLP
from caldera.blocks import MultiAggregator
from caldera.blocks import NodeBlock
from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data import GraphDataLoader
from caldera.data.utils import in_degree
from caldera.defaults import CalderaDefaults
from caldera.models import GraphCore
from caldera.models import GraphEncoder
from caldera.utils import deterministic_seed
from caldera.utils.nx import nx_iter_roots
from caldera.utils.tensor import to_one_hot

SEED = 0


class NamedNetwork:
    def __init__(self, name, network_func):
        self.name = name
        self.f = network_func

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class Networks:
    """Networks that will be used in the tests."""

    n = NamedNetwork

    linear_block = n(
        "linear",
        lambda: torch.nn.Sequential(
            torch.nn.Linear(5, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
        ),
    )

    mlp_block = n(
        "mlp",
        lambda: torch.nn.Sequential(
            Flex(MLP)(Flex.d(), 16), Flex(torch.nn.Linear)(Flex.d(), 1)
        ),
    )

    node_block = n(
        "node_block",
        lambda: torch.nn.Sequential(
            NodeBlock(Flex(MLP)(Flex.d(), 25, 25, layer_norm=False)),
            Flex(torch.nn.Linear)(Flex.d(), 1),
        ),
    )

    edge_block = n(
        "edge_block",
        lambda: torch.nn.Sequential(
            EdgeBlock(Flex(MLP)(Flex.d(), 25, 25, layer_norm=False)),
            Flex(torch.nn.Linear)(Flex.d(), 1),
        ),
    )

    global_block = n(
        "global_block",
        lambda: torch.nn.Sequential(
            GlobalBlock(Flex(MLP)(Flex.d(), 25, 25, layer_norm=False)),
            Flex(torch.nn.Linear)(Flex.d(), 1),
        ),
    )

    graph_encoder = n(
        "graph_encoder",
        lambda: GraphEncoder(
            EdgeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), 5, 5, layer_norm=False),
                    Flex(torch.nn.Linear)(Flex.d(), 1),
                )
            ),
            NodeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), 5, 5, layer_norm=False),
                    Flex(torch.nn.Linear)(Flex.d(), 1),
                )
            ),
            GlobalBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), 5, 5, layer_norm=False),
                    Flex(torch.nn.Linear)(Flex.d(), 1),
                )
            ),
        ),
    )

    def create_graph_core(pass_global_to_edge: bool, pass_global_to_node: bool):
        return GraphCore(
            AggregatingEdgeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), 5, 5, layer_norm=False),
                    Flex(torch.nn.Linear)(Flex.d(), 1),
                )
            ),
            AggregatingNodeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), 5, 5, layer_norm=False),
                    Flex(torch.nn.Linear)(Flex.d(), 1),
                ),
                edge_aggregator=Aggregator("add"),
            ),
            AggregatingGlobalBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), 5, 5, layer_norm=False),
                    Flex(torch.nn.Linear)(Flex.d(), 1),
                ),
                edge_aggregator=Aggregator("add"),
                node_aggregator=Aggregator("add"),
            ),
            pass_global_to_edge=pass_global_to_edge,
            pass_global_to_node=pass_global_to_node,
        )

    graph_core = n("graph_core", create_graph_core)

    def create_graph_core_multi_agg(
        pass_global_to_edge: bool, pass_global_to_node: bool
    ):
        agg = lambda: Flex(MultiAggregator)(Flex.d(), ["add", "mean", "max", "min"])

        return GraphCore(
            AggregatingEdgeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(
                        Flex.d(), 5, 5, layer_norm=True, activation=torch.nn.LeakyReLU
                    ),
                    Flex(torch.nn.Linear)(Flex.d(), 1),
                )
            ),
            AggregatingNodeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(
                        Flex.d(), 5, 5, layer_norm=True, activation=torch.nn.LeakyReLU
                    ),
                    Flex(torch.nn.Linear)(Flex.d(), 1),
                ),
                edge_aggregator=agg(),
            ),
            AggregatingGlobalBlock(
                torch.nn.Sequential(
                    Flex(MLP)(
                        Flex.d(), 5, 5, layer_norm=True, activation=torch.nn.LeakyReLU
                    ),
                    Flex(torch.nn.Linear)(Flex.d(), 1),
                ),
                edge_aggregator=agg(),
                node_aggregator=agg(),
            ),
            pass_global_to_edge=pass_global_to_edge,
            pass_global_to_node=pass_global_to_node,
        )

    graph_core_multi_agg = n("graph_core(multiagg)", create_graph_core_multi_agg)

    @staticmethod
    def reset(net: torch.nn.Module):
        def weight_reset(model):
            for layer in model.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        net.apply(weight_reset)


class DataModifier:
    """Methods to modify data before training."""

    def __init__(self, datalist):
        self.datalist = datalist

    @staticmethod
    def node_sum(batch: GraphBatch, copy=True):
        if copy:
            batch = batch.copy()
        batch.x = torch.cat([batch.x, batch.x.sum(axis=1, keepdim=True)], axis=1)
        return batch

    @staticmethod
    def edge_sum(batch: GraphBatch, copy=True):
        if copy:
            batch = batch.copy()
        batch.e = torch.cat([batch.e, batch.e.sum(axis=1, keepdim=True)], axis=1)
        return batch

    @staticmethod
    def global_sum(batch: GraphBatch, copy=True):
        if copy:
            batch = batch.copy()
        batch.g = torch.cat([batch.g, batch.g.sum(axis=1, keepdim=True)], axis=1)
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


class DataLoaders:
    """Data loaders for test."""

    @staticmethod
    def random_loader(data_size, batch_size):
        datalist = [GraphData.random(5, 5, 5) for _ in range(data_size)]
        return GraphDataLoader(datalist, batch_size)

    @staticmethod
    def _default_g(g: nx.DiGraph, global_key: str = None):
        for _, data in g.nodes(data=True):
            data["features"] = np.zeros((1,))
            data["target"] = np.zeros((1,))

        for _, _, data in g.edges(data=True):
            data["features"] = np.zeros((1,))
            data["target"] = np.zeros((1,))

        g.set_global({"features": np.zeros((1,)), "target": np.zeros((1,))}, global_key)
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
                ndata["features"] = to_one_hot(i, s)
                if i % 2 == 0:
                    target = np.array([0.5])
                else:
                    target = np.zeros(1)
                ndata["target"] = target

            input_data.append(GraphData.from_networkx(g, feature_key="features"))
            output_data.append(GraphData.from_networkx(g, feature_key="target"))

        return GraphDataLoader(
            list(zip(input_data, output_data)), batch_size=batch_size
        )

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
                edata["features"] = to_one_hot(i, s)
                if i % 2 == 0:
                    target = np.array([0.5])
                else:
                    target = np.zeros((1,))
                edata["target"] = target

            input_data.append(GraphData.from_networkx(g, feature_key="features"))
            output_data.append(GraphData.from_networkx(g, feature_key="target"))

        return GraphDataLoader(
            list(zip(input_data, output_data)), batch_size=batch_size
        )

    @classmethod
    def random_graph_red_black_global(cls, data_size, batch_size):
        input_data = []
        output_data = []
        s = 2
        for _ in range(data_size):
            g = nx.to_directed(nx.random_tree(10))
            cls._default_g(g)

            gdata = g.get_global()
            i = np.random.randint(0, 1, (1,))
            gdata["features"] = to_one_hot(i, s)
            if i % 2 == 0:
                target = np.array([0.5])
            else:
                target = np.zeros((1,))
            gdata["target"] = target

            input_data.append(GraphData.from_networkx(g, feature_key="features"))
            output_data.append(GraphData.from_networkx(g, feature_key="target"))

        return GraphDataLoader(
            list(zip(input_data, output_data)), batch_size=batch_size
        )

    @classmethod
    def est_density(cls, data_size, batch_size):
        input_data = []
        output_data = []
        s = 2
        for _ in range(data_size):
            n_size = np.random.randint(2, 20)
            g = nx.to_directed(nx.random_tree(n_size))
            cls._default_g(g)

            gdata = g.get_global()
            gdata["features"] = np.random.randn(1)
            gdata["target"] = np.array([nx.density(g)])

            input_data.append(GraphData.from_networkx(g, feature_key="features"))
            output_data.append(GraphData.from_networkx(g, feature_key="target"))

        return GraphDataLoader(
            list(zip(input_data, output_data)), batch_size=batch_size
        )

    @classmethod
    def in_degree(cls, data_size, batch_size):
        input_data = []
        output_data = []
        s = 2
        for _ in range(data_size):
            n_size = np.random.randint(2, 20)
            g = nx.to_directed(nx.random_tree(n_size))
            cls._default_g(g)

            for n, ndata in g.nodes(data=True):
                ndata["features"] = np.random.randn(1)
                ndata["target"] = np.array([in_degree(n)])

            input_data.append(GraphData.from_networkx(g, feature_key="features"))
            output_data.append(GraphData.from_networkx(g, feature_key="target"))

        return GraphDataLoader(
            list(zip(input_data, output_data)), batch_size=batch_size
        )

    @classmethod
    def boolean_network(cls, data_size, batch_size):

        input_data = []
        output_data = []
        for _ in range(data_size):
            n_size = np.random.randint(2, 20)
            tree = nx.random_tree(n_size)

            # randomize node directions
            g = nx.DiGraph()
            for n1, n2, edata in tree.edges(data=True):
                i = np.random.randint(2)
                if i % 2 == 0:
                    g.add_edge(n1, n2)
                else:
                    g.add_edge(n2, n1)
            cls._default_g(g)

            for n in nx_iter_roots(g):
                ndata = g.nodes[n]
                ndata["target"] = np.array([1.0])

            for n in nx.topological_sort(g):
                ndata = g.nodes[n]
                if "target" not in ndata:
                    incoming = []
                    for p in g.predecessors(n):
                        pdata = g.nodes[p]
                        incoming.append(pdata["target"])
                    incoming = np.concatenate(incoming)
                    i = incoming.max()
                    if i == 1:
                        o = np.array([0.0])
                    else:
                        o = np.array([1.0])
                    ndata["target"] = o

            input_data.append(GraphData.from_networkx(g, feature_key="features"))
            output_data.append(GraphData.from_networkx(g, feature_key="target"))

        return GraphDataLoader(
            list(zip(input_data, output_data)), batch_size=batch_size
        )

    @classmethod
    def sigmoid_circuit(cls, data_size, batch_size):
        import math

        def func(x):
            return 1 - 1.0 / (1 + math.exp(-x))

        input_data = []
        output_data = []
        for _ in range(data_size):
            n_size = np.random.randint(2, 20)
            tree = nx.random_tree(n_size)

            # randomize node directions
            g = nx.DiGraph()
            for n1, n2, edata in tree.edges(data=True):
                i = np.random.randint(2)
                if i % 2 == 0:
                    g.add_edge(n1, n2)
                else:
                    g.add_edge(n2, n1)
            cls._default_g(g)

            for n in nx_iter_roots(g):
                ndata = g.nodes[n]
                ndata["target"] = np.array(3.0)

            for n in nx.topological_sort(g):
                ndata = g.nodes[n]
                if "target" not in ndata:
                    incoming = []
                    for p in g.predecessors(n):
                        pdata = g.nodes[p]
                        incoming.append(pdata["target"])
                    incoming = np.concatenate(incoming)
                    i = incoming.sum()
                    o = func(i)
                    ndata["target"] = o

            input_data.append(GraphData.from_networkx(g, feature_key="features"))
            output_data.append(GraphData.from_networkx(g, feature_key="target"))

        return GraphDataLoader(
            list(zip(input_data, output_data)), batch_size=batch_size
        )


T = Tuple[Tuple[Tuple[Any, ...], Dict], torch.Tensor]


class DataGetter:
    """Methods to collect input, output from the loader."""

    @classmethod
    def get_node(cls, batch: GraphBatch) -> T:
        args = (batch.x[:, :-1],)
        kwargs = {}
        out = batch.x[:, -1:]
        return ((args, kwargs), out)

    @classmethod
    def get_edge(cls, batch: GraphBatch) -> T:
        args = (batch.e[:, :-1],)
        kwargs = {}
        out = batch.e[:, -1:]
        return ((args, kwargs), out)

    @classmethod
    def get_global(cls, batch: GraphBatch) -> T:
        args = (batch.g[:, :-1],)
        kwargs = {}
        out = batch.g[:, -1:]
        return ((args, kwargs), out)

    @classmethod
    def get_batch(cls, batch_tuple: Tuple[GraphBatch, GraphBatch]) -> T:
        args = (batch_tuple[0],)
        kwargs = {}
        out = batch_tuple[1]
        return ((args, kwargs), (out.e, out.x, out.g))


class NetworkTestCaseValidationError(Exception):
    pass


@contextmanager
def does_not_raise():
    yield


# TODO: model reset is not working
class NetworkTestCase:
    """A network test case."""

    def __init__(
        self,
        network: torch.nn.Module,
        modifier: Optional[Callable[[GraphBatch], Any]] = None,
        getter: Optional[Callable[[GraphBatch], Any]] = None,
        optimizer: Type[torch.optim.Optimizer] = None,
        criterion=None,
        loss_func: Callable = None,
        epochs: int = 20,
        batch_size: int = 100,
        data_size: int = 1000,
        loader: Optional[Callable[[int, int], GraphDataLoader]] = None,
        expectation: Callable = None,
        tags: Tuple[str, ...] = None,
        device: str = None,
    ):
        if expectation is None:
            expectation = does_not_raise()
        self.expectation = expectation
        self.tags = tags
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
        self.device = device
        if loader is None:
            loader = DataLoaders.random_loader
        self.loader_func = loader
        self.loader = self.loader_func(data_size, batch_size)
        self.optimizer = optimizer
        if criterion is None:
            criterion = torch.nn.MSELoss()
        if loss_func is not None:
            loss_func = functools.partial(loss_func, criterion, self.device)
        else:
            loss_func = criterion
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

    def eval(self, data_size):
        self.network.eval()
        with torch.no_grad():
            running_loss = 0.0
            for batch in self.loader_func(data_size, data_size):
                batch = self.to(batch)
                batch = self.modifier(batch)
                input, target = self.getter(batch)
                output = self.network(*input[0], **input[1])
                loss = self.loss_func(output, target)
                running_loss += loss.item()
            print("TARGET")
            print(target)
            print("OUTPUT")
            print(output)
        return running_loss

    def train(self):
        print("Training {}".format(self.network))
        self.reset()
        epochs = self.epochs
        net = self.network
        loader = self.loader
        optimizer = self.optimizer
        getter = self.getter
        modifier = self.modifier
        loss_func = self.loss_func

        # provide example
        self.provide_example()

        if optimizer is None:
            optimizer = optim.AdamW(net.parameters(), lr=1e-2)

        self.pre_train_validate()

        loss_arr = torch.zeros(epochs)
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            for batch in loader:
                batch = self.to(batch)
                batch = modifier(batch)
                input, target = getter(batch)

                optimizer.zero_grad()  # zero the gradient buffers
                output = net(*input[0], **input[1])

                for x, o, t in zip(["edge", "node", "global"], output, target):
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
            raise NetworkTestCaseValidationError(
                "Model did not train properly :(."
                "\n\tlosses: {} -> {}".format(self.losses[0], self.losses[-1])
            )

    def __str__(self):
        pass


@pytest.mark.parametrize(
    "loader_func",
    [
        DataLoaders.random_loader,
        DataLoaders.random_graph_red_black_nodes,
        DataLoaders.random_graph_red_black_edges,
        DataLoaders.est_density,
    ],
)
def test_loaders(loader_func):
    loader = loader_func(100, 20)
    for x in loader:
        assert x


def mse_tuple(criterion, device, a, b):
    loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    assert len(a) == len(b)
    for i, (_a, _b) in enumerate(zip(a, b)):
        assert _a.shape == _b.shape
        l = criterion(_a, _b)
        loss = loss + l
    return loss


def get_id(case):
    print(case.__class__)
    tokens = OrderedDict(
        {"id": None, "name": None, "loader": None, "expectation": None}
    )

    tokens["name"] = case["network"].name
    tokens["id"] = case.get("id", None)
    try:
        tokens["loader"] = case.get("loader", None).__name__
    except AttributeError:
        pass

    try:
        tokens["expectation"] = case.get("expectation", None)
    except AttributeError:
        pass

    return "-".join([str(v) for v in tokens.values() if v is not None])


@pytest.fixture
def network_case(request):
    def pop(d, k, default):
        if k in d:
            res = d[k]
            del d[k]
            return res
        return default

    params = dict(request.param)
    args = pop(params, "network_args", tuple())
    kwargs = pop(params, "network_kwargs", {})
    params["network"] = params["network"](*args, **kwargs)
    case = NetworkTestCase(**params)
    return case


cases = [
    dict(
        network=Networks.linear_block,
        modifier=DataModifier.node_sum,
        getter=DataGetter.get_node,
        tags=["block", "basic"],
    ),
    dict(
        network=Networks.mlp_block,
        modifier=DataModifier.node_sum,
        getter=DataGetter.get_node,
        tags=["block", "basic"],
    ),
    dict(
        network=Networks.node_block,
        modifier=DataModifier.node_sum,
        getter=DataGetter.get_node,
        tags=["block", "basic", "node"],
    ),
    dict(
        network=Networks.edge_block,
        modifier=DataModifier.edge_sum,
        getter=DataGetter.get_edge,
        tags=["block", "basic", "edge"],
    ),
    dict(
        network=Networks.global_block,
        modifier=DataModifier.global_sum,
        getter=DataGetter.get_global,
        tags=["block", "basic", "global"],
    ),
    dict(
        network=Networks.graph_encoder,
        loader=DataLoaders.random_graph_red_black_nodes,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["graph_encoder", "node"],
    ),  # randomly creates an input value, assigns 'red' or 'black' to nodes
    dict(
        network=Networks.graph_encoder,
        loader=DataLoaders.random_graph_red_black_edges,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["graph_encoder", "edge"],
    ),  # randomly creates an input value, assigns 'red' or 'black' to edges
    dict(
        network=Networks.graph_encoder,
        loader=DataLoaders.random_graph_red_black_global,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["graph_encoder", "global"],
    ),  # randomly creates an input value, assigns 'red' or 'black' to global
    dict(
        network=Networks.graph_encoder,
        loader=DataLoaders.est_density,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        expectation=pytest.raises(NetworkTestCaseValidationError),
        tags=["graph_encoder", "fail"],
    ),  # network cannot learn the density without connections between nodes and edges,
    dict(
        network=Networks.graph_core,
        network_kwargs={"pass_global_to_edge": True, "pass_global_to_node": True},
        loader=DataLoaders.est_density,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["graph_core", "global"],
    ),  # estimate the graph density using GraphCore
    dict(
        network=Networks.graph_core,
        network_kwargs={"pass_global_to_edge": False, "pass_global_to_node": True},
        loader=DataLoaders.est_density,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["graph_core", "global"],
    ),  # estimate the graph density using GraphCore
    dict(
        network=Networks.graph_core,
        network_kwargs={"pass_global_to_edge": True, "pass_global_to_node": False},
        loader=DataLoaders.est_density,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["graph_core", "global"],
    ),  # estimate the graph density using GraphCore
    dict(
        network=Networks.graph_core,
        network_kwargs={"pass_global_to_edge": False, "pass_global_to_node": False},
        loader=DataLoaders.est_density,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["graph_core", "global"],
    ),  # estimate the graph density using GraphCore
    dict(
        network=Networks.graph_core,
        network_kwargs={"pass_global_to_edge": True, "pass_global_to_node": True},
        loader=DataLoaders.in_degree,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["graph_core", "node"],
    ),  # estimate the graph density using GraphCore
    dict(
        network=Networks.graph_encoder,
        loader=DataLoaders.in_degree,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["graph_core", "node"],
        expectation=pytest.raises(NetworkTestCaseValidationError),
    ),  # estimate the graph density using GraphCore
    dict(
        network=Networks.graph_core,
        network_kwargs={"pass_global_to_edge": True, "pass_global_to_node": True},
        loader=DataLoaders.boolean_network,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["boolean_circuit"],
    ),  # estimate the graph density using GraphCore
    dict(
        network=Networks.graph_encoder,
        loader=DataLoaders.boolean_network,
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        tags=["boolean_circuit"],
        expectation=pytest.raises(NetworkTestCaseValidationError),
    ),  # estimate the graph density using GraphCore
    dict(
        network=Networks.graph_core,
        loader=DataLoaders.sigmoid_circuit,
        network_kwargs={"pass_global_to_edge": True, "pass_global_to_node": True},
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        epochs=100,
        tags=["sigmoid_circuit"],
    ),  # estimate the graph density using GraphCore
    dict(
        network=Networks.graph_core_multi_agg,
        loader=DataLoaders.sigmoid_circuit,
        network_kwargs={"pass_global_to_edge": True, "pass_global_to_node": True},
        getter=DataGetter.get_batch,
        loss_func=mse_tuple,
        epochs=100,
        tags=["sigmoid_circuit_(multiagg)"],
    ),  # estimate the graph density using GraphCore
]
# in degree
# average in degree
# a function of number of nodes, in degree
# boolean network that depends on multiple passes
# sigmoid circuit
# shortest _path example
visited_cases = set()


def parameterize_by_group(groups: Tuple[str, ...] = None) -> Callable:
    params = []
    for idx, p in enumerate(cases):
        if groups is None:
            params.append(p)
        else:
            for tag in p.get("tags", []):
                if tag in groups:
                    params.append(p)
                    visited_cases.add(idx)
                    break
    if not params:
        raise Exception("There are no cases with tags '{}'".format(groups))
    return pytest.mark.parametrize("network_case", params, ids=get_id, indirect=True)


def run_test_case(network_case, device):
    network_case.device = device
    with network_case.expectation:
        losses = network_case.train()
        print(losses)
        for p in network_case.network.parameters():
            assert device == str(p.device)
        network_case.post_train_validate()
    network_case.eval(20)
    return network_case


class TestTraining:
    @parameterize_by_group(["basic", "block"])
    def test_train_block(self, network_case, device):
        run_test_case(network_case, device)

    @parameterize_by_group(["graph_encoder"])
    def test_train_encoder(self, network_case, device):
        run_test_case(network_case, device)

    @parameterize_by_group(["graph_core"])
    def test_train_core(self, network_case, device):
        run_test_case(network_case, device)

    @parameterize_by_group(["boolean_circuit"])
    def test_train_boolean_circuit(self, network_case, device):
        run_test_case(network_case, device)

    @parameterize_by_group(["sigmoid_circuit"])
    def test_train_sigmoid_circuit(self, network_case, device):
        run_test_case(network_case, device)

    @parameterize_by_group(["sigmoid_circuit_(multiagg)"])
    def test_train_sigmoid_circuit_with_multi_agg(self, network_case, device):
        run_test_case(network_case, device)

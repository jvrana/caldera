import pytest
import networkx as nx
import random
import numpy as np
from collections import OrderedDict
from typing import Callable, Dict
from functools import reduce
import operator

##########################################
# Graph Generator Fixtures
##########################################


class GraphGenerator(object):
    @staticmethod
    def update_params(default: dict, params):
        new_params = OrderedDict(default)
        if isinstance(params, tuple):
            for x, (k, v) in zip(params, default.items()):
                new_params[k] = x
        elif isinstance(params, dict):
            new_params.update(params)
        else:
            key = list(default.keys())[0]
            new_params[key] = params
        return new_params

    @staticmethod
    def add_edge_data(g: nx.Graph, edge_data: Dict[str, Callable]):
        for n1, n2, edata in g.edges(data=True):
            for (k, v) in edge_data.items():
                edata[k] = v()
        return g

    @classmethod
    def init_graph(cls, g: nx.Graph, edge_data=None):
        if edge_data:
            cls.add_edge_data(g, edge_data)
        return g

    @classmethod
    def complete_graph(cls, param=None):
        params = OrderedDict(
            {
                "n_nodes": 10,
                "create_using": nx.DiGraph,
                "edge_data": {
                    "A": lambda: float(random.randint(0, 100)),
                    "B": lambda: np.random.uniform(0.75, 0.9),
                },
            }
        )
        if param is not None:
            params = cls.update_params(params, param)
        G = nx.complete_graph(params["n_nodes"], create_using=params["create_using"])
        cls.init_graph(G, edge_data=params["edge_data"])
        return G, params, param

    @classmethod
    def grid_graph(cls, param=None):
        params = OrderedDict(
            {
                "dims": [2, 3],
                "edge_data": {
                    "A": lambda: float(random.randint(0, 100)),
                    "B": lambda: np.random.uniform(0.75, 0.9),
                },
            }
        )
        if param is not None:
            params = cls.update_params(params, param)
        G = nx.grid_graph(params["dims"])
        cls.init_graph(G, edge_data=params["edge_data"])
        return G, params, param


@pytest.fixture
def graph_generator():
    return GraphGenerator()


@pytest.fixture(params=[(10, nx.DiGraph)])
def complete_graph(request):
    return GraphGenerator.complete_graph(request.param)


@pytest.fixture(params=[([2, 3],)])
def grid_graph(request):
    return GraphGenerator.grid_graph(request.param)


def get_indirect_params(request):
    data = {}
    for marker in request.node.own_markers:
        if marker.kwargs.get("indirect", False):
            data[marker.args[0]] = marker.args[1]
    return data


##########################################
# END
# Graph Generator Fixtures
##########################################


##########################################
# Test Graph Generator Fixtures
##########################################


class TestFixtures:
    def test_complete_graph(self, complete_graph):
        g, params, param = complete_graph
        assert isinstance(g, params["create_using"])
        assert g

    @pytest.mark.parametrize(
        "complete_graph",
        [
            (1, nx.DiGraph),
            (10, nx.DiGraph),
            # (1, ),
            # 10,
            # (100, nx.DiGraph, {'A': lambda: np.random.randint(0, 10)}),
            # (100, nx.Graph, {'A': lambda: np.random.randint(0, 10), 'C': lambda: 1})
        ],
        indirect=True,
    )
    def test_complete_graph_indirect(self, complete_graph, request):
        g, params, indirect_params = complete_graph
        for n1, n2, edata in g.edges(data=True):
            for k in params["edge_data"]:
                assert k in edata
        assert isinstance(g, params["create_using"])
        assert g.number_of_nodes() == indirect_params[0]

    def test_grid_graph(self, grid_graph):
        g, params, param = grid_graph
        assert isinstance(g, nx.Graph)

    @pytest.mark.parametrize(
        "grid_graph",
        [([2, 3],), ([10, 10], {"A": lambda: np.random.randint(0, 10)})],
        indirect=True,
    )
    def test_grid_graph_indirect(self, grid_graph):
        g, params, indirect_params = grid_graph
        assert isinstance(g, nx.Graph)
        assert g.number_of_nodes() == reduce(operator.mul, (indirect_params[0]))

    @pytest.mark.parametrize("fname", ["complete_graph", "grid_graph"])
    def test_graph_generator(self, fname, graph_generator):
        getattr(graph_generator, fname)()


##########################################
# END
# Test Graph Generator Fixtures
##########################################

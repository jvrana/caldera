import inspect

import networkx as nx
import numpy as np
import pytest

from caldera.transforms.networkx import NetworkxApplyToKey
from caldera.transforms.networkx import NetworkxAttachNumpyBool
from caldera.transforms.networkx import NetworkxAttachNumpyFeatures
from caldera.transforms.networkx import NetworkxAttachNumpyOneHot
from caldera.transforms.networkx import NetworkxFilterDataKeys
from caldera.transforms.networkx import NetworkxNodesToStr
from caldera.transforms.networkx import NetworkxSetDefaultFeature
from caldera.transforms.networkx import NetworkxToDirected
from caldera.transforms.networkx import NetworkxToUndirected
from caldera.transforms.networkx._base import NetworkxTransformBase


class TestTransformBase:
    class TransformTest(NetworkxTransformBase):
        def transform(self, x):
            return x

    def test_transform_from_list_to_list(self):
        t = self.TransformTest()
        x1 = [1, 2, 3]
        x2 = t(x1)
        assert isinstance(x2, list)
        assert x2 == [1, 2, 3]

    def test_transform_from_graph_to_graph(self):
        t = self.TransformTest()
        x1 = nx.Graph()
        x2 = t(x1)
        assert isinstance(x2, nx.Graph)

    def test_transform_from_iterable_to_iterable(self):
        t = self.TransformTest()
        x1 = iter([1, 2, 3])
        x2 = t(x1)
        assert not isinstance(x2, list)
        assert inspect.isgenerator(x2)
        assert list(x2) == [1, 2, 3]
        assert list(x2) == []

    def test_transform_tuple_to_tuple(self):
        t = self.TransformTest()
        x1 = tuple([1, 2, 3])
        x2 = t(x1)
        assert isinstance(x2, tuple)
        assert x2 == (1, 2, 3)


class TestApplyToKey:
    def test_(self):
        g = nx.DiGraph()
        g.add_node(1, features=2)
        t = NetworkxApplyToKey("features", node_func=lambda x: x * 10)
        t(g)
        print(g.nodes(data=True))
        assert g.nodes[1]["features"] == 20


def np_same(a, b):
    if not a.shape == b.shape:
        return False
    if not np.all(a == b):
        return False
    return True


class TestOneHot:
    def test_nodes_to_one_hot(self):
        g = nx.DiGraph()
        g.add_node(1, **{"features": True})
        g.add_node(2, **{"features": False})
        g.add_node(3, **{"features": False})
        classes = [True, False]

        transform = NetworkxAttachNumpyOneHot(
            "node", "features", "x", classes=classes, num_classes=10
        )

        transform(transform([g]))[0]

        a1 = np.zeros(10)
        a1[0] = 1
        a2 = np.zeros(10)
        a2[1] = 1

        a1 = np.hstack([a1, a1])
        a2 = np.hstack([a2, a2])

        assert np_same(g.nodes[1]["x"], a1)
        assert np_same(g.nodes[2]["x"], a2)
        assert np_same(g.nodes[3]["x"], a2)

        assert set(g.nodes[1]) == {"features", "x"}

    def test_one_hot_bool_class(self):
        g = nx.DiGraph()
        g.add_node(1, **{"features": True})
        g.add_node(2, **{"features": False})
        g.add_node(3, **{"features": False})

        transform = NetworkxAttachNumpyBool("node", "features", "x")

        transform(g)
        assert np_same(g.nodes[1]["x"], np.array([1]))
        assert np_same(g.nodes[2]["x"], np.array([0]))
        assert np_same(g.nodes[3]["x"], np.array([0]))
        #
        # transform(transform([g]))[0]
        #
        # a1 = np.zeros(10)
        # a1[0] = 1
        # a2 = np.zeros(10)
        # a2[1] = 1
        #
        # a1 = np.hstack([a1, a1])
        # a2 = np.hstack([a2, a2])
        #
        # assert np_same(g.nodes[1]["x"], a1)
        # assert np_same(g.nodes[2]["x"], a2)
        # assert np_same(g.nodes[3]["x"], a2)
        #
        # assert set(g.nodes[1]) == {"features", "x"}

    def test_edges_to_one_hot(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, **{"features": True})
        g.add_edge(2, 3, **{"features": False})
        g.add_edge(3, 4, **{"features": False})
        classes = [True, False]

        transform = NetworkxAttachNumpyOneHot(
            "edge", "features", "x", classes=classes, num_classes=10
        )

        transform(transform([g]))

        a1 = np.zeros(10)
        a1[0] = 1
        a2 = np.zeros(10)
        a2[1] = 1

        a1 = np.hstack([a1, a1])
        a2 = np.hstack([a2, a2])

        assert np_same(g.edges[(1, 2)]["x"], a1)
        assert np_same(g.edges[(2, 3)]["x"], a2)
        assert np_same(g.edges[(3, 4)]["x"], a2)

    def test_nodes_to_one_hot_missing_attribute_raises_key_error(self):
        g = nx.DiGraph()
        g.add_node(1, **{"features": True})
        g.add_node(2, **{"features": False})
        g.add_node(3)

        transform = NetworkxAttachNumpyOneHot(
            "node", "features", "x", classes=[True, False], num_classes=10
        )

        with pytest.raises(KeyError):
            transform([g])

    def test_edges_to_one_hot_missing_attribute_raises_key_error(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, **{"features": True})
        g.add_edge(2, 3, **{"features": False})
        g.add_edge(3, 4)

        transform = NetworkxAttachNumpyOneHot(
            "edge", "features", "x", classes=[True, False], num_classes=10
        )

        with pytest.raises(KeyError):
            transform([g])


class TestCollectIterables:
    #
    def test_vstack_nodes(self):
        g = nx.DiGraph()
        g.add_node(1, features=[0, 1, 2])
        g.add_node(2, features=np.array([1, 2, 3]))
        g.add_node(3, features=np.array([1, 3]))

        transform = NetworkxAttachNumpyFeatures(
            "node", "features", "x", join_func="vstack"
        )
        g = transform(transform([g]))[0]
        print(g.nodes[1])

        assert np_same(g.nodes[1]["x"], np.array([[0, 1, 2], [0, 1, 2]]))
        assert np_same(g.nodes[2]["x"], np.array([[1, 2, 3], [1, 2, 3]]))

    def test_vstack_edges(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, features=[0, 1, 2])
        g.add_edge(2, 3, features=np.array([1, 2, 3]))

        transform = NetworkxAttachNumpyFeatures(
            "edge", "features", "x", join_func="vstack"
        )
        g = transform(transform([g]))[0]

        assert np_same(g.edges[(1, 2)]["x"], np.array([[0, 1, 2], [0, 1, 2]]))
        assert np_same(g.edges[(2, 3)]["x"], np.array([[1, 2, 3], [1, 2, 3]]))

    def test_vstack_globals(self):
        graphs = [nx.DiGraph(), nx.DiGraph()]
        graphs[0].set_global({"features": [0, 1, 2]})
        graphs[1].set_global({"features": np.array([1, 2, 3])})

        transform = NetworkxAttachNumpyFeatures(
            "global", "features", "x", join_func="vstack"
        )
        graphs = transform(transform(graphs))

        print(graphs[0].get_global())
        assert np_same(graphs[0].get_global()["x"], np.array([[0, 1, 2], [0, 1, 2]]))
        assert np_same(graphs[1].get_global()["x"], np.array([[1, 2, 3], [1, 2, 3]]))

    def test_hstack_nodes(self):
        g = nx.DiGraph()
        g.add_node(1, features=[0, 1, 2])
        g.add_node(2, features=np.array([1, 2, 3]))
        g.add_node(3, features=np.array([1, 3]))

        transform = NetworkxAttachNumpyFeatures("node", "features", "x")
        g = transform(transform([g]))[0]
        print(g.nodes[1])

        assert np_same(g.nodes[1]["x"], np.array([0, 1, 2, 0, 1, 2]))
        assert np_same(g.nodes[2]["x"], np.array([1, 2, 3, 1, 2, 3]))

    def test_hstack_edges(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, features=[0, 1, 2])
        g.add_edge(2, 3, features=np.array([1, 2, 3]))

        transform = NetworkxAttachNumpyFeatures("edge", "features", "x")
        g = transform(transform([g]))[0]

        assert np_same(g.edges[(1, 2)]["x"], np.array([0, 1, 2, 0, 1, 2]))
        assert np_same(g.edges[(2, 3)]["x"], np.array([1, 2, 3, 1, 2, 3]))


class TestToDirected:
    @pytest.mark.parametrize("args", [tuple(), (nx.OrderedDiGraph,)])
    def test_to_directed(self, args):
        g = nx.Graph()
        g.add_edge(1, 2)
        transform = NetworkxToDirected(*args)
        assert g.edges[(1, 2)] == {}
        assert g.edges[(2, 1)] == {}

        dg = transform(g)
        if args:
            assert isinstance(dg, args[0])
        else:
            assert isinstance(dg, nx.DiGraph)
        assert dg.edges[(1, 2)] == {}
        assert dg.edges[(2, 1)] == {}

    @pytest.mark.parametrize("args", [tuple(), (nx.OrderedGraph,)])
    def test_to_undirected(self, args):
        g = nx.DiGraph()
        g.add_edge(1, 2)
        transform = NetworkxToUndirected(*args)
        assert g.edges[(1, 2)] == {}

        udg = transform(g)
        if args:
            assert isinstance(udg, args[0])
        else:
            assert isinstance(udg, nx.Graph)
            assert not isinstance(udg, nx.DiGraph)
        assert udg.edges[(1, 2)] == {}
        assert udg.edges[(2, 1)] == {}


class TestAddDefault:
    def test_add_node_default(self):
        g = nx.DiGraph()
        g.add_node(1, other=1)
        transform = NetworkxSetDefaultFeature(node_default={"feature": True})
        g = transform(g)
        assert g.nodes[1] == {"feature": True, "other": 1}

    def test_add_edge_default(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, other=1)
        transform = NetworkxSetDefaultFeature(edge_default={"feature": True})
        g = transform(g)
        assert g.edges[(1, 2)] == {"feature": True, "other": 1}

    def test_add_global_default(self):
        g = nx.DiGraph()
        g.get_global()["other"] = 3
        g.add_node(1)
        transform = NetworkxSetDefaultFeature(global_default={"feature": True})
        g = transform(g)
        assert g.get_global() == {"feature": True, "other": 3}

    def test_add_default(self):
        g = nx.DiGraph()
        g.add_edge(1, 2)
        transform = NetworkxSetDefaultFeature(
            node_default={"n": True}, edge_default={"e": False}, global_default={"g": 5}
        )
        g = transform(g)
        assert g.nodes[1] == {"n": True}
        assert g.nodes[2] == {"n": True}
        assert g.edges[(1, 2)] == {"e": False}
        assert g.get_global() == {"g": 5}


class TestNodeToStr:
    def test_node_to_str(self):
        g = nx.DiGraph()
        g.add_edge(1, 2)
        transform = NetworkxNodesToStr()
        g = transform(g)
        assert g.number_of_nodes() == 2


class TestFilterDataKeys:
    def test_filer_node_keys(self):
        g = nx.DiGraph()
        g.add_node(1, x=1, y=2)
        g.add_node(2, x=2)
        g.add_node(3, y=2)
        t = NetworkxFilterDataKeys(node_keys=["x"])

        g = t([g])[0]
        assert g.nodes[1] == {"x": 1}
        assert g.nodes[2] == {"x": 2}
        assert g.nodes[3] == {}

    def test_filer_edge_keys(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, x=1, y=2)
        g.add_edge(2, 3, x=2)
        g.add_edge(3, 4, y=2)
        t = NetworkxFilterDataKeys(edge_keys=["x"])

        g = t([g])[0]
        assert g.edges[(1, 2)] == {"x": 1}
        assert g.edges[(2, 3)] == {"x": 2}
        assert g.edges[(3, 4)] == {}

    def test_filer_global_keys(self):
        g = nx.DiGraph()
        g.set_global({"x": [1, 2, 3], "y": 2})
        t = NetworkxFilterDataKeys(global_keys=["x"])
        g2 = t([g])[0]
        assert g2.get_global() == {"x": [1, 2, 3]}

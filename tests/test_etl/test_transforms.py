import networkx as nx
import numpy as np
import pytest

from caldera.etl.transforms import NetworkxAttachNumpyFeatures
from caldera.etl.transforms import NetworkxAttachNumpyOneHot
from caldera.etl.transforms._base import NetworkxTransformBase


class TestTransformBase:
    def test_transform_from_list_to_list(self):
        pytest.fail("")

    def test_transform_from_iterable_to_iterable(self):
        pytest.fail("")


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

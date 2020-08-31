import networkx as nx
import numpy as np
import pytest

from caldera.utils.nx.convert import add_default_edge_data
from caldera.utils.nx.convert import add_default_node_data
from caldera.utils.nx.convert import nx_collect_edges_hstack
from caldera.utils.nx.convert import nx_collect_edges_vstack
from caldera.utils.nx.convert import nx_collect_nodes_hstack
from caldera.utils.nx.convert._nx_np_features import nx_collect_nodes_vstack
from caldera.utils.nx.convert._nx_np_features import nx_collect_one_hot_edges_hstack
from caldera.utils.nx.convert._nx_np_features import nx_collect_one_hot_globals_hstack
from caldera.utils.nx.convert._nx_np_features import nx_collect_one_hot_globals_vstack
from caldera.utils.nx.convert._nx_np_features import nx_collect_one_hot_nodes_hstack


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
        result = nx_collect_one_hot_nodes_hstack(g, "features", "x", classes, n=10)
        result = nx_collect_one_hot_nodes_hstack(g, "features", "x", classes, n=10)

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
        result = nx_collect_one_hot_edges_hstack(g, "features", "x", classes, n=10)
        result = nx_collect_one_hot_edges_hstack(g, "features", "x", classes, n=10)

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

        with pytest.raises(KeyError):
            result = nx_collect_one_hot_nodes_hstack(
                g, "features", "x", [True, False], n=10
            )

    def test_edges_to_one_hot_missing_attribute_raises_key_error(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, **{"features": True})
        g.add_edge(2, 3, **{"features": False})
        g.add_edge(3, 4)

        with pytest.raises(KeyError):
            result = nx_collect_one_hot_edges_hstack(
                g, "features", "x", [True, False], n=10
            )

    def test_add_default_node_data(self):
        g = nx.DiGraph()
        g.add_node(3)
        g.add_edge(1, 3, features=5)
        add_default_node_data(g, {"features": 2})
        assert g.nodes[3] == {"features": 2}
        assert g.edges[(1, 3)] == {"features": 5}

    def test_add_default_edge_data(self):
        g = nx.DiGraph()
        g.add_edge(1, 2)
        g.add_node(1, features=5)
        add_default_edge_data(g, {"features": 2})
        assert g.edges[(1, 2)] == {"features": 2}
        assert g.nodes[1] == {"features": 5}

    def test_nodes_to_one_hot_missing_attribute_with_add_default(self):
        g = nx.DiGraph()
        g.add_node(1, **{"features": True})
        g.add_node(2, **{"features": False})
        g.add_node(3)

        add_default_node_data(g, {"features": 2})

        result = nx_collect_one_hot_nodes_hstack(
            g, "features", "x", [True, False, 2], n=5
        )
        assert np_same(g.nodes[3]["x"], np.array([0, 0, 1, 0, 0]))

    def test_edges_to_one_hot_missing_attribute_with_add_default(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, **{"features": True})
        g.add_edge(2, 3, **{"features": False})
        g.add_edge(3, 4)

        add_default_edge_data(g, {"features": 2})

        result = nx_collect_one_hot_edges_hstack(
            g, "features", "x", [True, False, 2], n=5
        )
        assert np_same(g.edges[(3, 4)]["x"], np.array([0, 0, 1, 0, 0]))

    # TODO: test global
    def test_graphs_global_to_one_hot_hstack(self):
        graphs = [nx.DiGraph(), nx.DiGraph()]
        graphs[0].data = {"features": True}
        graphs[1].data = {"features": False}

        nx_collect_one_hot_globals_hstack(
            graphs[0], "features", "x", [True, False], n=3
        )
        nx_collect_one_hot_globals_hstack(
            graphs[0], "features", "x", [True, False], n=3
        )
        nx_collect_one_hot_globals_hstack(
            graphs[1], "features", "x", [True, False], n=3
        )
        nx_collect_one_hot_globals_hstack(
            graphs[1], "features", "x", [True, False], n=3
        )

        assert np_same(graphs[0].data["x"], np.array([1, 0, 0, 1, 0, 0]))
        assert np_same(graphs[1].data["x"], np.array([0, 1, 0, 0, 1, 0]))

    def test_graphs_global_to_one_hot_vstack(self):
        graphs = [nx.DiGraph(), nx.DiGraph()]
        graphs[0].data = {"features": True}
        graphs[1].data = {"features": False}

        nx_collect_one_hot_globals_vstack(
            graphs[0], "features", "x", [True, False], n=3
        )
        nx_collect_one_hot_globals_vstack(
            graphs[0], "features", "x", [True, False], n=3
        )
        nx_collect_one_hot_globals_vstack(
            graphs[1], "features", "x", [True, False], n=3
        )
        nx_collect_one_hot_globals_vstack(
            graphs[1], "features", "x", [True, False], n=3
        )

        assert np_same(graphs[0].data["x"], np.array([[1, 0, 0]] * 2))
        assert np_same(graphs[1].data["x"], np.array([[0, 1, 0]] * 2))


class TestCollectIterables:
    #
    def test_vstack_nodes(self):
        g = nx.DiGraph()
        g.add_node(1, features=[0, 1, 2])
        g.add_node(2, features=np.array([1, 2, 3]))
        g.add_node(3, features=np.array([1, 3]))

        nx_collect_nodes_vstack(g, "features", "x")
        nx_collect_nodes_vstack(g, "features", "x")
        print(g.nodes[1])

        assert np_same(g.nodes[1]["x"], np.array([[0, 1, 2], [0, 1, 2]]))
        assert np_same(g.nodes[2]["x"], np.array([[1, 2, 3], [1, 2, 3]]))

    def test_vstack_edges(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, features=[0, 1, 2])
        g.add_edge(2, 3, features=np.array([1, 2, 3]))

        nx_collect_edges_vstack(g, "features", "x")
        nx_collect_edges_vstack(g, "features", "x")

        assert np_same(g.edges[(1, 2)]["x"], np.array([[0, 1, 2], [0, 1, 2]]))
        assert np_same(g.edges[(2, 3)]["x"], np.array([[1, 2, 3], [1, 2, 3]]))

    # def test_vstack_globals(self):
    #     graphs = [
    #         nx.DiGraph(),
    #         nx.DiGraph()
    #     ]
    #     graphs[0].data = {'features': [0, 1, 2]}
    #     graphs[1].data = {'features': np.array([1, 2, 3])}
    #
    #     nxs_vstack_global(graphs, 'features', 'x')
    #     nxs_vstack_global(graphs, 'features', 'x')
    #
    #     assert np_same(graphs[0].data['x'], np.array([[0, 1, 2], [0, 1, 2]]))
    #     assert np_same(graphs[1].data['x'], np.array([[1, 2, 3], [1, 2, 3]]))

    def test_hstack_nodes(self):
        g = nx.DiGraph()
        g.add_node(1, features=[0, 1, 2])
        g.add_node(2, features=np.array([1, 2, 3]))
        g.add_node(3, features=np.array([1, 3]))

        nx_collect_nodes_hstack(g, "features", "x")
        nx_collect_nodes_hstack(g, "features", "x")
        print(g.nodes[1])

        assert np_same(g.nodes[1]["x"], np.array([0, 1, 2, 0, 1, 2]))
        assert np_same(g.nodes[2]["x"], np.array([1, 2, 3, 1, 2, 3]))

    def test_hstack_edges(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, features=[0, 1, 2])
        g.add_edge(2, 3, features=np.array([1, 2, 3]))

        nx_collect_edges_hstack(g, "features", "x")
        nx_collect_edges_hstack(g, "features", "x")

        assert np_same(g.edges[(1, 2)]["x"], np.array([0, 1, 2, 0, 1, 2]))
        assert np_same(g.edges[(2, 3)]["x"], np.array([1, 2, 3, 1, 2, 3]))

    #
    # def test_hstack_globals(self):
    #     graphs = [
    #         nx.DiGraph(),
    #         nx.DiGraph()
    #     ]
    #     graphs[0].data = {'features': [0, 1, 2]}
    #     graphs[1].data = {'features': np.array([1, 2, 3])}
    #
    #     nxs_hstack_global(graphs, 'features', 'x')
    #     nxs_hstack_global(graphs, 'features', 'x')
    #
    #     assert np_same(graphs[0].data['x'], np.array([0, 1, 2, 0, 1, 2]))
    #     assert np_same(graphs[1].data['x'], np.array([1, 2, 3, 1, 2, 3]))

import networkx as nx
from caldera.data.utils._nx_conversion import nx_collect_one_hot_nodes, nx_collect_one_hot_edges, add_default_edge_data, add_default_node_data
import numpy as np
import pytest


def test_nodes_to_one_hot():
    g = nx.DiGraph()
    g.add_node(1, **{'features': True})
    g.add_node(2, **{'features': False})
    g.add_node(3, **{'features': False})
    result = nx_collect_one_hot_nodes(g, 'features', 'x', 10)
    result = nx_collect_one_hot_nodes(g, 'features', 'x', 10)

    a1 = np.zeros(10)
    a1[1] = 1
    a2 = np.zeros(10)
    a2[0] = 1

    a1 = np.hstack([a1, a1])
    a2 = np.hstack([a2, a2])

    assert np.all(g.nodes[1]['x'] == a1)
    assert np.all(g.nodes[2]['x'] == a2)
    assert np.all(g.nodes[3]['x'] == a2)


def test_edges_to_one_hot():
    g = nx.DiGraph()
    g.add_edge(1, 2, **{'features': True})
    g.add_edge(2, 3, **{'features': False})
    g.add_edge(3, 4, **{'features': False})
    result = nx_collect_one_hot_edges(g, 'features', 'x', 10)
    result = nx_collect_one_hot_edges(g, 'features', 'x', 10)

    a1 = np.zeros(10)
    a1[1] = 1
    a2 = np.zeros(10)
    a2[0] = 1

    a1 = np.hstack([a1, a1])
    a2 = np.hstack([a2, a2])

    assert np.all(g.edges[(1, 2)]['x'] == a1)
    assert np.all(g.edges[(2, 3)]['x'] == a2)
    assert np.all(g.edges[(3, 4)]['x'] == a2)


def test_nodes_to_one_hot_missing_attribute_raises_key_error():
    g = nx.DiGraph()
    g.add_node(1, **{'features': True})
    g.add_node(2, **{'features': False})
    g.add_node(3)

    with pytest.raises(KeyError):
        result = nx_collect_one_hot_nodes(g, 'features', 'x', 10)


def test_edges_to_one_hot_missing_attribute_raises_key_error():
    g = nx.DiGraph()
    g.add_edge(1, 2, **{'features': True})
    g.add_edge(2, 3, **{'features': False})
    g.add_edge(3, 4)

    with pytest.raises(KeyError):
        result = nx_collect_one_hot_edges(g, 'features', 'x', 10)


def test_nodes_to_one_hot_missing_attribute_with_add_default():
    g = nx.DiGraph()
    g.add_node(1, **{'features': True})
    g.add_node(2, **{'features': False})
    g.add_node(3)

    with pytest.raises(KeyError):
        result = nx_collect_one_hot_nodes(g, 'features', 'x', 10)


def test_edges_to_one_hot_missing_attribute_with_add_default():
    g = nx.DiGraph()
    g.add_edge(1, 2, **{'features': True})
    g.add_edge(2, 3, **{'features': False})
    g.add_edge(3, 4)

    add_default_edge_data(g, {'features': 'Third'})

    result = nx_collect_one_hot_edges(g, 'features', 'x', 10)
    result = nx_collect_one_hot_edges(g, 'features', 'x', 10)

    a1 = np.zeros(10)
    a1[0] = 1
    a2 = np.zeros(10)
    a2[1] = 1
    a3 = np.zeros(10)
    a3[2] = 1

    a1 = np.hstack([a1, a1])
    a2 = np.hstack([a2, a2])
    a3 = np.hstack([a3, a3])

    assert np.all(g.edges[(1, 2)]['x'] == a1)
    assert np.all(g.edges[(2, 3)]['x'] == a2)
    assert np.all(g.edges[(3, 4)]['x'] == a3)
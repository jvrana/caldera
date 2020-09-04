import networkx as nx
import pytest

from caldera.utils.nx import nx_copy


def add_data(g):
    g.add_node(1)
    g.add_node(2, x=5)
    g.add_edge(1, 2, y=6)
    g.add_edge(2, 3, z=[])


def assert_graph_data(g1, g2):
    assert g1 is not g2
    assert g2.nodes[1] == {}
    assert g2.nodes[2] == {"x": 5}
    assert g2.edges[(1, 2)] == {"y": 6}
    assert g2.edges[(2, 3)] == {"z": []}
    assert g2.nodes[2] is not g1.nodes[2]
    assert g2.edges[(2, 3)] is not g1.edges[(2, 3)]


@pytest.mark.parametrize("do_deepcopy", [True, False])
def test_nx_copy_with_deepcopy(do_deepcopy):
    g = nx.Graph()
    g2 = nx.DiGraph()
    add_data(g)
    nx_copy(g, g2, deepcopy=do_deepcopy)
    assert_graph_data(g, g2)
    assert (g2.edges[(2, 3)]["z"] is g.edges[(2, 3)]["z"]) != do_deepcopy


def test_nx_copy_with_none():
    g = nx.Graph()
    add_data(g)
    g2 = nx_copy(g, None)
    assert_graph_data(g, g2)


def test_nx_copy_with_class():
    g = nx.Graph()
    add_data(g)
    g2 = nx_copy(g, nx.OrderedDiGraph)
    assert isinstance(nx.OrderedDiGraph, type) and issubclass(
        nx.OrderedDiGraph, nx.Graph
    )
    assert isinstance(g2, nx.OrderedDiGraph)
    assert_graph_data(g, g2)


def test_nx_copy_node_transform():
    g = nx.Graph()
    g.add_node(1)
    g.add_node(2)
    g.add_edge(1, 2, f=4)
    g.add_edge(2, 3, f=5)

    def node_transform(nodes):
        for n, ndata in nodes:
            yield str(n), ndata

    g2 = nx_copy(g, None, node_transform=node_transform)
    assert g2.number_of_nodes() == 3
    assert g2.number_of_edges() == 2
    assert "1" in g2
    assert "2" in g2
    assert 1 not in g2
    assert 2 not in g2
    assert g2.edges[("1", "2")] == {"f": 4}
    assert g2.edges[("2", "3")] == {"f": 5}


def test_nx_copy_edge_transform():
    g = nx.Graph()
    g.add_node(1)
    g.add_node(2)
    g.add_edge(1, 2, f=4)
    g.add_edge(2, 3, f=5)
    g.add_edge(4, 5)

    assert g.number_of_edges() == 3
    assert g.number_of_nodes() == 5

    def edge_transform(edges):
        for n1, n2, edata in edges:
            if n1 != 4:
                yield n1, n2, {"f": 8}

    g2 = nx_copy(g, None, edge_transform=edge_transform)
    assert g2.number_of_nodes() == 5
    assert g2.number_of_edges() == 2
    assert g2.edges[(1, 2)] == {"f": 8}
    assert g2.edges[(2, 3)] == {"f": 8}


def test_nx_copy_global_transform():
    g = nx.Graph()
    g.add_node(1)
    g.add_node(2)
    g.add_edge(1, 2, f=4)
    g.add_edge(2, 3, f=5)
    g.add_edge(4, 5)
    g.get_global()["f"] = 8
    assert g.number_of_edges() == 3
    assert g.number_of_nodes() == 5

    def global_transform(g):
        for _, gdata in g:
            gdata["x"] = 4
            yield _, gdata

    g2 = nx_copy(g, None, global_transform=global_transform)
    assert g2.get_global() == {"x": 4, "f": 8}

import networkx as nx
from caldera.data.utils._nx_conversion import nx_collect_one_hot_nodes, nx_collect_one_hot_edges


def test_nodes_to_one_hot():
    g = nx.DiGraph()
    g.add_node(1, **{'features': True})
    g.add_node(2, **{'features': False})
    g.add_node(3, **{'features': False})
    result = nx_collect_one_hot_nodes(g, 'features', 'x', 10)
    result = nx_collect_one_hot_nodes(g, 'features', 'x', 10)
    for n, ndata in g.nodes(data=True):
        print(ndata)
    assert False


def test_edges_to_one_hot():
    g = nx.DiGraph()
    g.add_node(1, **{'features': True})
    g.add_node(2, **{'features': False})
    g.add_node(3, **{'features': False})
    nx_collect_one_hot_edges(g, 'features', 'x', 2)
    assert False

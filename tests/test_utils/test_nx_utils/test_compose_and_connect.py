import uuid

import networkx as nx
import pytest

from caldera.utils import functional as Fn
from caldera.utils.nx.generators import chain_graph
from caldera.utils.nx.generators import compose_and_connect
from caldera.utils.nx.generators import random_graph
from caldera.utils.nx.generators import unique_chain_graph


def test_unique_chain_graph():
    g = unique_chain_graph(10, nx.DiGraph)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 9


@pytest.mark.parametrize(("n1", "n2", "density"), [(10, 10, 0.05)])
def test_compare_and_connect_directed(n1, n2, density):
    g = unique_chain_graph(n1, nx.DiGraph)
    h = unique_chain_graph(n2, nx.DiGraph)
    z = nx.compose(g, h)
    assert z.number_of_edges() == g.number_of_edges() + h.number_of_edges()
    j = compose_and_connect(g, h, density)
    n_new_edges = j.number_of_edges() - (g.number_of_edges() + h.number_of_edges())
    assert n_new_edges == int(density * (n1 * n2)) * 2


@pytest.mark.parametrize(("n1", "n2", "density"), [(10, 10, 0.05)])
def test_compare_and_connect_undirected(n1, n2, density):
    g = unique_chain_graph(n1, nx.Graph)
    h = unique_chain_graph(n2, nx.Graph)
    z = nx.compose(g, h)
    assert z.number_of_edges() == g.number_of_edges() + h.number_of_edges()
    j = compose_and_connect(g, h, density)
    n_new_edges = j.number_of_edges() - (g.number_of_edges() + h.number_of_edges())
    assert n_new_edges == int(density * (n1 * n2))

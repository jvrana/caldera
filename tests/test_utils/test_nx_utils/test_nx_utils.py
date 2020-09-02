import functools

import networkx as nx
import pytest
from dictdiffer import diff

from caldera.utils.nx import nx_copy
from caldera.utils.nx import nx_is_directed
from caldera.utils.nx import nx_is_undirected
from caldera.utils.nx import nx_to_directed
from caldera.utils.nx import nx_to_undirected


@pytest.mark.parametrize("dp", [True, False])
def test_nx_copy(dp):
    g = nx.Graph()
    g.add_node(1)
    g.add_node(2, x=5)
    g.add_edge(1, 2, y=6)
    g.add_edge(2, 3, z=[])
    g2 = nx.Graph()
    nx_copy(g, g2, deepcopy=dp)
    assert g2.nodes[1] == {}
    assert g2.nodes[2] == {"x": 5}
    assert g2.edges[(1, 2)] == {"y": 6}
    assert g2.edges[(2, 3)] == {"z": []}
    assert g2.edges[(2, 3)] is not g.edges[(2, 3)]
    assert (g2.edges[(2, 3)]["z"] is g.edges[(2, 3)]["z"]) != dp


undirected_types = [nx.Graph, nx.MultiGraph, nx.OrderedGraph, nx.OrderedMultiGraph]
directed_types = [
    nx.DiGraph,
    nx.MultiDiGraph,
    nx.OrderedDiGraph,
    nx.OrderedMultiDiGraph,
]


def mark_xfail(params, **kwargs):
    return [pytest.param(p, marks=pytest.mark.xfail(**kwargs)) for p in params]


def create_graph(cls):
    g = cls()
    g.add_node(1, x=6)
    g.add_edge(1, 2)
    g.add_edge(2, 3, x=5, y=[])
    return g


def _graph_compare(g1, g2, f):
    diffs = diff(f(g1), f(g2))
    return list(diffs)


# TODO: tests for deep copy and copy


diff_nodes = functools.partial(_graph_compare, f=lambda x: dict(x.nodes(data=True)))
diff_edges = functools.partial(
    _graph_compare, f=lambda x: {(e[0], e[1]): e[2] for e in x.edges(data=True)}
)


class TestDirectedAndUndirected:
    class TestIsDirected:
        @pytest.mark.parametrize("c", directed_types)
        def test_is_directed(self, c):
            g = c()
            assert nx_is_directed(g)

        @pytest.mark.parametrize("c", undirected_types)
        def test_is_not_directed(self, c):
            g = c()
            assert not nx_is_directed(g)

    class TestIsUndirected:
        @pytest.mark.parametrize("c", undirected_types)
        def test_is_undirected(self, c):
            g = c()
            assert nx_is_undirected(g)

        @pytest.mark.parametrize("c", directed_types)
        def test_is_not_undirected(self, c):
            g = c()
            assert not nx_is_undirected(g)

    class TestConvertToDirected:
        @pytest.mark.parametrize("c1", undirected_types, ids=lambda x: str(x))
        @pytest.mark.parametrize(
            "c2",
            directed_types + mark_xfail(directed_types, strict=True),
            ids=lambda x: str(x),
        )
        def test_undirected_to_directed(self, c1, c2):
            g1 = create_graph(c1)
            if c2 is None:
                g2 = nx_to_directed(g1)
            else:
                g2 = nx_to_directed(g1, graph_class=c2)
            assert isinstance(g2, c2)
            assert g1.number_of_edges() * 2 == g2.number_of_edges()
            assert not diff_nodes(g1, g2)
            assert diff_edges(g1, g2)

        @pytest.mark.parametrize("c1", directed_types, ids=lambda x: str(x))
        @pytest.mark.parametrize(
            "c2",
            directed_types + mark_xfail(undirected_types, strict=True),
            ids=lambda x: str(x),
        )
        def test_directed_to_directed(self, c1, c2):
            g1 = create_graph(c1)
            if c2 is None:
                g2 = nx_to_directed(g1)
            else:
                g2 = nx_to_directed(g1, graph_class=c2)
            assert isinstance(g2, c2)
            assert g1.number_of_edges() == g2.number_of_edges()
            assert not diff_nodes(g1, g2)
            assert not diff_edges(g1, g2)

    class TestConvertToUndirected:
        @pytest.mark.parametrize("c1", directed_types, ids=lambda x: str(x))
        @pytest.mark.parametrize(
            "c2",
            undirected_types + mark_xfail(directed_types, strict=True),
            ids=lambda x: str(x),
        )
        def test_directed_to_undirected(self, c1, c2):
            g1 = create_graph(c1)
            if c2 is None:
                g2 = nx_to_undirected(g1)
            else:
                g2 = nx_to_undirected(g1, graph_class=c2)
            assert isinstance(g2, c2)
            assert not diff_nodes(g1, g2)
            assert not diff_edges(g1, g2)

        @pytest.mark.parametrize("c1", undirected_types, ids=lambda x: str(x))
        @pytest.mark.parametrize(
            "c2",
            undirected_types + mark_xfail(directed_types, strict=True),
            ids=lambda x: str(x),
        )
        def test_undirected_to_undirected(self, c1, c2):
            g1 = create_graph(c1)
            if c2 is None:
                g2 = nx_to_undirected(g1)
            else:
                g2 = nx_to_undirected(g1, graph_class=c2)
            assert isinstance(g2, c2)
            assert not diff_nodes(g1, g2)
            assert not diff_edges(g1, g2)

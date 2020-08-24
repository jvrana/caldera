import pytest


class TestDijkstras:
    def test_simple_path(self):
        G = nx.path_graph(4)

        add_data(G, 0, 1, 100, 0.5)
        add_data(G, 1, 2, 100, 0.5)
        add_data(G, 2, 3, 100, 0.5)

        path_length, path = sympy_multisource_dijkstras(
            G, [0], "weight / eff", accumulators={"eff": "product"}
        )

        assert path_length[0] == 0.0
        assert path_length[1] == 200.0
        assert path_length[2] == 100 * 2 / 0.5 ** 2
        assert path_length[3] == 100 * 3 / 0.5 ** 3

    def test_simple_path(self):
        G = nx.path_graph(4)
        G.add_edge(0, 1, weight=100, eff=0.5)
        G.add_edge(1, 2, weight=100, eff=0.5)
        G.add_edge(2, 3, weight=100, eff=0.5)
        G.add_edge(3, 4, weight=100, eff=0.5)
        G.add_edge(0, 5, weight=200, eff=0.75)
        G.add_edge(5, 6, weight=200, eff=0.75)
        G.add_edge(6, 7, weight=200, eff=0.75)
        G.add_edge(7, 4, weight=200, eff=0.75)

        path_length, path = sympy_multisource_dijkstras(
            G, [0], "weight / eff", accumulators={"eff": "product"}
        )

        assert path_length[0] == 0.0
        assert path_length[1] == 100 * 1 / 0.5 ** 1
        assert path_length[2] == 100 * 2 / 0.5 ** 2
        assert path_length[3] == 100 * 3 / 0.5 ** 3
        assert path_length[4] == 200 * 4 / 0.75 ** 4
        assert path_length[5] == 200 * 1 / 0.75 ** 1
        assert path_length[6] == 200 * 2 / 0.75 ** 2
        assert path_length[7] == 200 * 3 / 0.75 ** 3

        assert path[3] == [0, 1, 2, 3]
        assert path[4] == [0, 5, 6, 7, 4]
        assert path[7] == [0, 5, 6, 7]

    def test_simple_path_source_to_target(self):
        G = nx.path_graph(4)
        G.add_edge(0, 1, weight=100, eff=0.5)
        G.add_edge(1, 2, weight=100, eff=0.5)
        G.add_edge(2, 3, weight=100, eff=0.5)
        G.add_edge(3, 4, weight=100, eff=0.5)
        G.add_edge(0, 5, weight=200, eff=0.75)
        G.add_edge(5, 6, weight=200, eff=0.75)
        G.add_edge(6, 7, weight=200, eff=0.75)
        G.add_edge(7, 4, weight=200, eff=0.75)

        path_length, path = sympy_dijkstras(
            g=G, source=0, target=4, f="weight / eff", accumulators={"eff": "product"}
        )
        assert path_length == 200 * 4 / 0.75 ** 4
        assert path == [0, 5, 6, 7, 4]

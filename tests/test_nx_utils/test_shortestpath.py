import networkx as nx
from caldera.utils.nx import multisource_dijkstras
from caldera.utils.nx import PathSymbol, PathNpSum, PathNpProduct
import numpy as np


class TestDijkstras:
    def test_simple_path(self, graph_generator):
        G = nx.path_graph(4)
        graph_generator.add_edge_data(
            G,
            {
                "weight": lambda: np.random.uniform(1, 10),
                "eff": lambda: np.random.uniform(0.7, 0.9),
            },
        )

        path_length, path = multisource_dijkstras(
            g=G,
            func=lambda w, e: w / e,
            symbols=[PathSymbol("weight", PathNpSum), PathSymbol("eff", PathNpProduct)],
            sources=[0],
        )

    def test_simple_path2(self):
        G = nx.path_graph(4)
        G.add_edge(0, 1, weight=100, eff=0.5)
        G.add_edge(1, 2, weight=100, eff=0.5)
        G.add_edge(2, 3, weight=100, eff=0.5)
        G.add_edge(3, 4, weight=100, eff=0.5)
        G.add_edge(0, 5, weight=200, eff=0.75)
        G.add_edge(5, 6, weight=200, eff=0.75)
        G.add_edge(6, 7, weight=200, eff=0.75)
        G.add_edge(7, 4, weight=200, eff=0.75)

        path_length, path = multisource_dijkstras(
            g=G,
            func=lambda w, e: w / e,
            symbols=[PathSymbol("weight", PathNpSum), PathSymbol("eff", PathNpProduct)],
            sources=[0],
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

        path_length, path = multisource_dijkstras(
            g=G,
            func=lambda w, e: w / e,
            symbols=[PathSymbol("weight", PathNpSum), PathSymbol("eff", PathNpProduct)],
            sources=[0],
            target=4,
        )
        assert path_length == 200 * 4 / 0.75 ** 4
        assert path == [0, 5, 6, 7, 4]

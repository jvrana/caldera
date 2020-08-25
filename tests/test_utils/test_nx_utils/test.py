import random

import networkx as nx
import numpy as np
import pytest

from caldera.utils.nx._all_pairs_shortest_path import floyd_warshall2
from caldera.utils.nx._all_pairs_shortest_path import PathMax
from caldera.utils.nx._all_pairs_shortest_path import PathMin
from caldera.utils.nx._all_pairs_shortest_path import PathMul
from caldera.utils.nx._all_pairs_shortest_path import PathSum
from caldera.utils.nx._all_pairs_shortest_path import PathSymbol


class TestSympyFloydWarshall:
    @pytest.mark.parametrize("key", ["weight", "cost"])
    @pytest.mark.parametrize("as_symbol", [True, False])
    @pytest.mark.parametrize(
        "complete_graph", [1, 100, 200, (100, nx.Graph)], indirect=True
    )
    def test_floyd_warshall(self, key, complete_graph, as_symbol):
        g, params, indirect_params = complete_graph
        nodelist = list(g.nodes())
        for n1, n2, edata in g.edges(data=True):
            for k in list(edata):
                del edata[k]
            edata[key] = random.randint(1, 10)
        expected = nx.floyd_warshall_numpy(g, nodelist=nodelist, weight=key)

        if as_symbol:
            symbols = [PathSymbol(key, PathNpSum)]
        else:
            symbols = [key]

        x = floyd_warshall2(g=g, symbols=symbols)

        assert np.allclose(x, expected)

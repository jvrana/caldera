from pprint import pprint

import networkx as nx
import numpy as np

from caldera.data import GraphData
from caldera.testing import add_default
from caldera.testing import annotate_shortest_path
from caldera.utils._tools import _resolve_range
from caldera.utils.functional import Functional as Fn
from caldera.utils.nx import nx_copy
from caldera.utils.nx import nx_to_directed
from caldera.utils.nx.convert import nx_collect_features
from caldera.utils.nx.convert._tools import feature_info
from caldera.utils.nx.generators import _uuid_chain
from caldera.utils.nx.generators import chain_graph
from caldera.utils.nx.generators import compose_and_connect
from caldera.utils.nx.generators import random_graph


def test_feature_info():
    g = random_graph((30, 50), density=(0.01, 0.3))
    add_default(g, node_data={"features": True, "target": False})
    feature_info(g)


def generate_shorest_path_example(n_nodes, density, path_length):
    d = _resolve_range(density)
    l = _resolve_range(path_length)
    path = list(_uuid_chain(l))
    h = chain_graph(path, nx.Graph)
    g = random_graph(n_nodes, density=d)
    graph = compose_and_connect(g, h, d)

    annotate_shortest_path(
        graph,
        True,
        True,
        source_key="source",
        target_key="target",
        path_key="shortest_path",
        source=path[0],
        target=path[-1],
    )

    # 1. add defaults
    add_default(
        graph, node_data={"source": False, "target": False, "shortest_path": False}
    )
    add_default(graph, edge_data={"shortest_path": False})

    # # 2. collect features
    nx_collect_features(
        graph, "node", "source", "_features", encoding="onehot", classes=[False, True]
    )
    nx_collect_features(
        graph, "node", "target", "_features", encoding="onehot", classes=[False, True]
    )

    # nx_collect_features(graph, 'node', 'shortest_path', '_target', encoding='onehot', classes=[False, True])
    nx_collect_features(
        graph,
        "edge",
        "shortest_path",
        "_target",
        encoding="onehot",
        classes=[False, True],
    )
    #

    # 3. make all node keys strings so that they are sortable
    copied_graph = nx_copy(
        graph, None, node_transform=Fn.map_each(lambda x: (str(x[0]), x[1]))
    )

    # 4. to directed
    copied_graph = nx_to_directed(copied_graph)

    # 5. fill in missing features
    GraphData.from_networkx(copied_graph, feature_key="_features")
    GraphData.from_networkx(copied_graph, feature_key="_target")

    return copied_graph


def test_():
    generate_shorest_path_example(100, 0.01, 10)


def test_shortest_path():
    g = random_graph((30, 50), density=(0.01, 0.3))
    annotate_shortest_path(g)
    pprint(feature_info(g))
    nx_collect_features(
        g,
        "node",
        from_key="source",
        to_key="_target",
        encoding="onehot",
        classes=[False, True],
    )
    nx_collect_features(
        g,
        "node",
        from_key="target",
        to_key="_target",
        encoding="onehot",
        classes=[False, True],
    )
    nx_collect_features(
        g,
        "edge",
        from_key="shortest_path",
        to_key="_target",
        encoding="onehot",
        classes=[False, True],
    )
    nx_collect_features(
        g,
        "node",
        from_key="shortest_path",
        to_key="_target",
        encoding="onehot",
        classes=[False, True],
    )
    pprint(feature_info(g))

    g = nx_to_directed(g)
    data = GraphData.from_networkx(g, feature_key="_target")
    assert data
    print(data)

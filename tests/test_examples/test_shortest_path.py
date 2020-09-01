from pprint import pprint

from caldera.data import GraphData
from caldera.testing import add_default
from caldera.testing import annotate_shortest_path
from caldera.testing import feature_info
from caldera.testing import random_graph
from caldera.utils.nx import nx_to_directed
from caldera.utils.nx import nx_to_undirected
from caldera.utils.nx.convert import nx_collect_features


def test_feature_info():
    g = random_graph((30, 50), density=(0.01, 0.3))
    add_default(g, node_data={"features": True, "target": False})
    feature_info(g)


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
    print()
    pprint(feature_info(g))

    import networkx as nx

    pprint(nx.info(g))
    g = nx_to_directed(g)
    GraphData.from_networkx(g, feature_key="_target")
    import networkx as nx

    pprint(nx.info(g))

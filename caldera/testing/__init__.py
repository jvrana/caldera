import networkx as nx

from .check_back_prop import check_back_prop
from caldera.transforms.networkx import NetworkxSetDefaultFeature
from caldera.utils.nx.generators import random_node


def annotate_shortest_path(
    g: nx.Graph,
    annotate_nodes: bool = True,
    annotate_edges: bool = True,
    source_key: str = "source",
    target_key: str = "target",
    path_key: str = "shortest_path",
    source: str = None,
    target: str = None,
    weight: str = "weight",
) -> nx.Graph:
    """

    :param g:
    :param annotate_nodes:
    :param annotate_edges:
    :param source_key:
    :param target_key:
    :param path_key:
    :param source:
    :param target:
    :return:
    """
    if not annotate_edges and not annotate_nodes:
        raise ValueError("Must annotate either nodes or edges (or both)")
    source = source or random_node(g)
    target = target or random_node(g)

    g.nodes[source][source_key] = True
    g.nodes[target][target_key] = True

    try:
        path = nx.shortest_path(g, source=source, target=target, weight=weight)
    except nx.NetworkXNoPath:
        path = []

    NetworkxSetDefaultFeature(
        node_default={target_key: False, source_key: False, path_key: False},
        edge_default={path_key: False},
    )

    if annotate_nodes:
        for n in path:
            g.nodes[n][path_key] = True

    if annotate_edges:
        for n1, n2 in nx.utils.pairwise(path):
            g.edges[(n1, n2)][path_key] = True


# def draw_shortest_path(g):
#     pprint(feature_info(g))
#     fig = plt.figure(figsize=(3, 3))
#     ax = fig.gca()
#     ax.axis('off')
#     nodelist = list(g.nodes)
#
#     node_color = []
#     for n in nodelist:
#         node_color.append(g.nodes[n]['shortest_path'])
#
#     edge_list = []
#     edge_color = []
#     for n1, n2, edata in g.edges(data=True):
#         edge_list.append((n1, n2))
#         edge_color.append(edata['shortest_path'])
#
#     pos = nx.layout.spring_layout(g)
#     nx.draw_networkx_nodes(g, pos=pos, node_color=node_color, node_size=10, ax=ax)
#     nx.draw_networkx_edges(g, pos=pos, width=0.5, edge_color=edge_color, ax=ax)

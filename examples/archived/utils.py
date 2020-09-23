# import random
#
# import networkx as nx
#
#
# # TODO: add long traversal
# def random_graph(n, e=None, d=None):
#     """Create a random graph."""
#     n = random.randint(*n)
#     if e is None:
#         d = random.random() * d[1] + d[0]
#         e = int((d * n * (n - 1)) / 2)
#     else:
#         e = random.randint(*e)
#     e = max(1, e)
#     return nx.generators.dense_gnm_random_graph(n, e)
#
#
# def annotate_shortest_path(g):
#     source, target = np.random.choice(list(g.nodes), size=(2,))
#
#     g.nodes[source]["source"] = True
#     g.nodes[target]["target"] = True
#     try:
#         traversal = nx.shortest_path(g, source=nodes[0], target=nodes[1])
#     except nx.NetworkXNoPath:
#         traversal = []
#
#     for n in traversal:
#         g.nodes[n]["shortest_path"] = False
#     for n, ndata in g.nodes(data=True):
#         ndata["shortest_path"] = target
#
#     for n1, n2, edata in g.edges(data=True):
#         edata["shortest_path"] = False
#
#     for n1, n2 in nx.utils.pairwise(traversal):
#         g[n1][n2]["shortest_path"] = True
#
#
# def cat_property(from_key, to_key):
#     pass
#
#
# def from_property_graph_to_graph_data():
#     pass

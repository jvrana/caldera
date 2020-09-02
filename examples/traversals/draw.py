# g = random_graph((100, 150), d=(0.01, 0.03), e=None)
# annotate_shortest_path(g)
# # nx.draw(g)
# pos = nx.layout.spring_layout(g)
# nodelist = list(g.nodes)
# node_color = []
# for n in nodelist:
#     node_color.append(g.nodes[n]['target'][0])
# edge_list = []
# edge_color = []
# for n1, n2, edata in g.edges(data=True):
#     edge_list.append((n1, n2))
#     edge_color.append(edata['target'][0])
# print(node_color)
# nx.draw_networkx_edges(g, pos=pos, width=0.5, edge_color=edge_color)
# nx.draw_networkx_nodes(g, pos=pos, node_color=node_color, node_size=10)

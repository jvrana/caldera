# flake8: noqa
import collections

import networkx as nx
import numpy as np
import pylab as plt


def sorted_center(iterable, reverse=False, key=None, select=None):
    centered = []
    s = sorted(iterable, key=key, reverse=reverse)
    front = False
    for v in s:
        if not front:
            centered.append(v)
        else:
            centered.insert(0, v)
        front = not front
    if select:
        return [select(_x) for _x in centered]
    return centered


class GraphPlotter:
    def __init__(self, graph, ax=None, pos=None):
        if ax is None:
            ax = plt.figure(figsize=(3, 3)).gca()
            ax.axis("off")
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.0)

        self._ax = ax
        self._graph = graph
        if pos is None:
            self._pos = dict()
            self.spring_layout()
        else:
            self._pos = pos

    @property
    def _base_draw_kwargs(self):
        return dict(G=self._graph, ax=self._ax, pos=self._pos)

    def topological_sort(self):
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()

        x = xlim[1] - xlim[0]
        y = ylim[1] - ylim[0]
        xspacer = x * 0.05
        yspacer = y * 0.05

        new_xlim = (xlim[0] + xspacer, xlim[1] - xspacer)
        new_ylim = (ylim[0] + yspacer, ylim[1] - yspacer)
        pos = self._topological_sort(self._graph, xlim=new_xlim, ylim=new_ylim)
        self._pos = pos

    @staticmethod
    def _roots_and_leaves(G, include_cycles=False):
        roots = [k for k, v in G.in_degree(G.nodes) if v == 0]
        leaves = [k for k, v in G.out_degree(G.nodes) if v == 0]

        if include_cycles:
            for c in nx.simple_cycles(G):
                outdegree = list(G.out_degree(c))
                outdegree.sort(key=lambda x: x[1])
                leaves.append(outdegree[0][0])

                indegree = list(G.out_degree(c))
                indegree.sort(key=lambda x: x[1])
                roots.append(indegree[0][0])
        return roots, leaves

    @classmethod
    def _get_roots(cls, G, include_cycles=False):
        return cls._roots_and_leaves(G, include_cycles=include_cycles)[0]

    @classmethod
    def _get_leaves(cls, G, include_cycles=False):
        return cls._roots_and_leaves(G, include_cycles=include_cycles)[1]

    @staticmethod
    def _get_subgraphs(graph):
        """Get independent subgraphs."""
        node_list = list(graph.nodes)
        subgraphs = []
        while len(node_list) > 0:
            node = node_list[-1]
            subgraph = nx.bfs_tree(to_undirected(graph), node)
            for n in subgraph.nodes:
                node_list.remove(n)
            subgraphs.append(graph.subgraph(subgraph.nodes))
        return subgraphs

    @classmethod
    def _topological_sort(cls, G, xlim=None, ylim=None):
        if xlim is None:
            xlim = [0.05, 0.95]
        if ylim is None:
            ylim = [0.05, 0.95]

        rts = cls._get_roots(G, include_cycles=True)
        max_depth = {}
        for root in rts:
            depths = nx.single_source_shortest_path_length(G, root)
            for n, d in depths.items():
                max_depth[n] = max(max_depth.get(n, d), d)

        by_depth = collections.OrderedDict()

        for node, depth in max_depth.items():
            by_depth.setdefault(depth, [])
            by_depth[depth].append(node)

        # center nodes with highest degree
        for depth, nodes in by_depth.items():
            centered = sorted_center(
                list(G.degree(nodes)),
                key=lambda x: x[1],
                reverse=True,
                select=lambda x: x[0],
            )
            by_depth[depth] = centered

        # push roots 'up' so they are not stuck on layer one
        for root in rts:
            successors = list(G.successors(root))
            if len(successors) > 0:
                min_depth = min([max_depth[s] for s in successors])
                max_depth[root] = min_depth - 1

        # assign positions

        y_min_max = xlim
        x_min_max = ylim
        max_width = max([len(layer) for layer in by_depth.values()])

        y_step = (y_min_max[1] - y_min_max[0]) / (max(by_depth.keys()) + 1)
        x_step = (x_min_max[1] - x_min_max[0]) / (max_width)
        positions = {}

        for depth in sorted(by_depth):
            y = y_step * depth + y_min_max[0]
            node_ids = by_depth[depth]
            w = len(node_ids)
            delta_w = max_width - w
            for i, n in enumerate(node_ids):
                x_offset = delta_w * x_step / 2.0
                x = x_min_max[0] + x_offset + i * x_step
                positions[n] = (x, y)
        return positions

    def spring_layout(self, **kwargs):
        pos = nx.spring_layout(self._graph, **kwargs)
        self._pos.update(pos)
        return self._pos

    @property
    def nodes(self):
        return self._graph.nodes

    @property
    def edges(self):
        return self._graph.edges

    def node_attrs(self):
        keys = set()
        for _, node_data in self._graph.nodes(data=True):
            keys.update(set(node_data.keys()))
        attrs = {}
        for _, node_data in self._graph.nodes(data=True):
            for k in keys:
                attrs.setdefault(k, list())
                attrs[k].append(node_data.get(k, None))
        return attrs

    def edge_attrs(self):
        keys = set()
        for _, _, edge_data in self._graph.edges(data=True):
            keys.update(set(edge_data.keys()))
        attrs = {}
        for _, _, edge_data in self._graph.edges(data=True):
            for k in keys:
                attrs.setdefault(k, list())
                attrs[k].append(edge_data.get(k, None))
        return attrs

    def map_edge_attrs(self, attrs, source, target):
        vals = [
            self._normalize(self.edge_attrs()[attr], source, target) for attr in attrs
        ]
        return dict(zip(attrs, vals))

    def map_node_attrs(self, attrs, source, target):
        vals = [
            self._normalize(self.node_attrs()[attr], source, target) for attr in attrs
        ]
        return dict(zip(attrs, vals))

    def _normalize(arr, source, target):
        x = source[1] - source[0]
        y = target[1] - target[0]
        return [_x / x * y + target[0] for _x in arr]

    def _make_draw_kwargs(self, **kwargs):
        kwargs.update(self._base_draw_kwargs)
        return kwargs

    def _draw(self, draw_function, zorder=None, **kwargs):
        draw_kwargs = self._make_draw_kwargs(**kwargs)
        collection = draw_function(**draw_kwargs)
        if collection is not None and zorder is not None:
            try:
                # This is for compatibility with older matplotlib.
                collection.set_zorder(zorder)
            except AttributeError:
                # This is for compatibility with newer matplotlib.
                collection[0].set_zorder(zorder)
        return collection

    def draw_nodes(self, **kwargs):
        """Useful kwargs: nodelist, node_size, node_color, linewidths."""
        if (
            "node_color" in kwargs
            and isinstance(kwargs["node_color"], collections.Sequence)
            and len(kwargs["node_color"]) in {3, 4}
            and not isinstance(
                kwargs["node_color"][0], (collections.Sequence, np.ndarray)
            )
        ):
            num_nodes = len(kwargs.get("nodelist", self.nodes))
            kwargs["node_color"] = np.tile(
                np.array(kwargs["node_color"])[None], [num_nodes, 1]
            )
        return self._draw(nx.draw_networkx_nodes, **kwargs)

    def draw_edges(self, **kwargs):
        """Useful kwargs: edgelist, width."""
        return self._draw(nx.draw_networkx_edges, **kwargs)

    def draw_graph(
        self,
        node_size=200,
        node_color=(0.4, 0.8, 0.4),
        node_linewidth=1.0,
        edge_width=1.0,
    ):

        node_border_color = (0.0, 0.0, 0.0, 1.0)

        # Plot nodes.
        self.draw_nodes(
            nodelist=self.nodes,
            node_size=node_size,
            node_color=node_color,
            linewidths=node_linewidth,
            edgecolors=node_border_color,
            zorder=20,
        )
        # Plot edges.
        self.draw_edges(edgelist=self.edges, width=edge_width, zorder=10)


g = nx.balanced_tree(2, 2)
g = nx.to_directed(g)

for e in g.edges:
    g.edges[e[0], e[1]]["weight"] = 1

ax = plt.figure(figsize=(3, 3)).gca()
ax.axis("off")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)

plotter = GraphPlotter(g, ax=ax)
plotter.topological_sort()

# plotter.draw_graph(node_color=(0, 0, 0), edge_width=plotter.edge_attrs()['weight'])

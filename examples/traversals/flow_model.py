from caldera import gnn
from torch import nn
from caldera._future.flow import Flow


class GraphCore(Flow):

    def __init__(self, node, edge, glob):
        super().__init__()
        self.node = node
        self.edge = edge
        self.glob = glob

        self.register_feed(lambda data: data.x, self.node)
        self.register_feed(lambda data: data.e, self.edge)
        self.register_feed(lambda data: data.g, self.glob)

        self.register_map(self.glob, self.edge, lambda data: data.edge_idx)
        self.register_map(self.node, self.edge, lambda data: data.edges[0])
        self.register_map(self.node, self.edge, lambda data: data.edges[1])
        self.register_map(self.glob, self.node, lambda data: data.node_idx)


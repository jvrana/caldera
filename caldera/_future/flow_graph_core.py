from caldera import gnn
from torch import nn
from caldera._future.flow import Flow
from caldera.data import GraphBatch


class GraphCore(Flow):

    def __init__(self):
        super().__init__()
        self.node = gnn.Flex(gnn.Dense)(..., 8, 8)
        self.edge = gnn.Flex(gnn.Dense)(..., 8, 8)
        self.glob = gnn.Flex(gnn.Dense)(..., 8, 8)

        self.register_connection(lambda data: data.x, self.node)
        self.register_connection(lambda data: data.e, self.edge)
        self.register_connection(lambda data: data.g, self.glob)

        self.register_connection(self.edge, self.node_to_edge_agg)

    def forward(self, data):
        x = self.propogate(self.node, data)
        e = self.propogate(self.edge, data)
        g = self.propogate(self.glob, data)
        return data.new_like(x, e, g)


core = GraphCore()

core(GraphBatch.random_batch(10, 5, 4, 3))
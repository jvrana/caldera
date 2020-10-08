from caldera import gnn
from torch import nn
from caldera._future.flow import Flow
from caldera.data import GraphBatch
from typing import List


class GraphEncoder(Flow):

    def __init__(self,
                 node: nn.Module,
                 edge: nn.Module,
                 glob: nn.Module
                 ):
        super().__init__()
        self.node = node
        self.edge = edge
        self.glob = glob

        self.register_feed(lambda data: data.x, self.node)
        self.register_feed(lambda data: data.e, self.edge)
        self.register_feed(lambda data: data.g, self.glob)

    def forward(self, data):
        x = self.propogate(self.node, data)
        e = self.propogate(self.edge, data)
        g = self.propogate(self.glob, data)
        return data.new_like(x, e, g)


class GraphCore(GraphEncoder):

    def __init__(self,
                 node: nn.Module,
                 edge: nn.Module,
                 glob: nn.Module
                 ):
        super().__init__(node, edge, glob)
        self.edge_to_node_agg = gnn.Aggregator('add')
        self.node_to_glob_agg = gnn.Aggregator('add')
        self.edge_to_glob_agg = gnn.Aggregator('add')

        self.register_map(self.glob, self.edge, lambda data: data.edge_idx)
        self.register_map(self.node, self.edge, lambda data: data.edges[0])
        self.register_map(self.node, self.edge, lambda data: data.edges[1])
        self.register_map(self.glob, self.node, lambda data: data.node_idx)

        self.register_aggregation(
            self.edge,
            self.edge_to_node_agg,
            self.node,
            lambda data: data.edges[1],
            lambda data: data.x.shape[0]
        )

        self.register_aggregation(
            self.edge,
            self.edge_to_glob_agg,
            self.glob,
            lambda data: data.edge_idx,
            lambda data: data.g.shape[0]
        )

        self.register_aggregation(
            self.node,
            self.node_to_glob_agg,
            self.glob,
            lambda data: data.node_idx,
            lambda data: data.g.shape[0]
        )

    def forward(self, data):
        x = self.propogate(self.node, data)
        e = self.propogate(self.edge, data)
        g = self.propogate(self.glob, data)
        return data.new_like(x, e, g)


class GraphProcess(nn.Module):

    def __init__(self, encoder, core, decoder, out):
        super().__init__()
        self.encoder = encoder
        self.core = core
        self.decoder = encoder
        self.out = out

    def forward(self, data):
        latent0 = self.encoder(data)
        data = latent0
        out_arr = []
        for step in range(5):
            data = self.core(latent0.cat(data))
            data = self.decoder(data)
            out_data = self.out(data)
            out_arr.append(out_data)
        return out_arr

process = GraphProcess(
    GraphEncoder(
        gnn.Flex(gnn.Dense)(..., 8),
        gnn.Flex(gnn.Dense)(..., 8),
        gnn.Flex(gnn.Dense)(..., 8),
    ),
    GraphCore(
        gnn.Flex(gnn.Dense)(..., 8),
        gnn.Flex(gnn.Dense)(..., 8),
        gnn.Flex(gnn.Dense)(..., 8),
    ),
    GraphCore(
        gnn.Flex(gnn.Dense)(..., 8),
        gnn.Flex(gnn.Dense)(..., 8),
        gnn.Flex(gnn.Dense)(..., 8),
    ),
    GraphCore(
        gnn.Flex(gnn.Dense)(..., 1),
        gnn.Flex(gnn.Dense)(..., 1),
        gnn.Flex(gnn.Dense)(..., 1),
    )
)

process(GraphBatch.random_batch(10, 5, 4, 3))
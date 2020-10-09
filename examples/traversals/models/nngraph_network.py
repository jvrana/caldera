from torch import nn

from caldera import gnn
from caldera._experimental.nngraph import NNGraph
from caldera.data import GraphBatch
from caldera.data import GraphTuple
from examples.traversals.configuration import NetConfig


class Encoder(NNGraph):
    def __init__(self, config):
        super().__init__()
        config = config.encode
        self.add_node(gnn.Flex(gnn.Dense)(..., config.node.size), "node")
        self.add_node(gnn.Flex(gnn.Dense)(..., config.edge.size), "edge")
        self.add_node(gnn.Flex(gnn.Dense)(..., config.glob.size), "glob")
        self.add_edge(lambda data: data.x, "node")
        self.add_edge(lambda data: data.e, "edge")
        self.add_edge(lambda data: data.g, "glob")

    def forward(self, data: GraphBatch):
        with self.run():
            x = self.propogate("node", data)
            e = self.propogate("edge", data)
            g = self.propogate("glob", data)
        return GraphBatch(
            x, e, g, edges=data.edges, node_idx=data.node_idx, edge_idx=data.edge_idx
        )


class Core(NNGraph):
    def __init__(self, config: NetConfig):
        super().__init__()
        config = config.core
        self.add_node(
            gnn.Flex(gnn.Dense)(..., *[config.node.size] * config.node.depth), "node"
        )
        self.add_node(
            gnn.Flex(gnn.Dense)(..., *[config.edge.size] * config.edge.depth), "edge"
        )
        self.add_node(
            gnn.Flex(gnn.Dense)(..., *[config.glob.size] * config.glob.depth), "glob"
        )
        self.add_edge(lambda data: data.x, "node")
        self.add_edge(lambda data: data.e, "edge")
        self.add_edge(lambda data: data.g, "glob")

        self.add_edge(lambda data: data.x, "edge", lambda data: data.edges[0])
        self.add_edge(lambda data: data.x, "edge", lambda data: data.edges[1])
        self.add_edge(lambda data: data.g, "edge", lambda data: data.edge_idx)
        self.add_edge(lambda data: data.g, "node", lambda data: data.node_idx)

        self.add_edge(
            "edge",
            "node",
            indexer=lambda data: data.edges[1],
            aggregation=gnn.Aggregator("add"),
            size=lambda data: data.x.shape[0],
        )
        self.add_edge(
            "edge",
            "glob",
            indexer=lambda data: data.edge_idx,
            aggregation=gnn.Aggregator("add"),
            size=lambda data: data.g.shape[0],
        )
        self.add_edge(
            "node",
            "glob",
            indexer=lambda data: data.node_idx,
            aggregation=gnn.Aggregator("add"),
            size=lambda data: data.g.shape[0],
        )

    def forward(self, data: GraphBatch):
        with self.run():
            x = self.propogate("node", data)
            e = self.propogate("edge", data)
            g = self.propogate("glob", data)
        return GraphBatch(
            x, e, g, edges=data.edges, node_idx=data.node_idx, edge_idx=data.edge_idx
        )


class OutTransform(NNGraph):
    def __init__(self, config: NetConfig):
        super().__init__()
        config = config.out
        self.add_node(
            nn.Sequential(
                gnn.Flex(nn.Linear)(..., config.node.size),
                getattr(nn, config.node.activation)(),
            ),
            "node",
        )
        self.add_node(
            nn.Sequential(
                gnn.Flex(nn.Linear)(..., config.edge.size),
                getattr(nn, config.edge.activation)(),
            ),
            "edge",
        )
        self.add_node(
            nn.Sequential(
                gnn.Flex(nn.Linear)(..., config.glob.size),
                getattr(nn, config.glob.activation)(),
            ),
            "glob",
        )

        self.add_edge(lambda data: data.x, "node")
        self.add_edge(lambda data: data.e, "edge")
        self.add_edge(lambda data: data.g, "glob")

    def forward(self, data: GraphBatch):
        with self.run():
            x = self.propogate("node", data)
            e = self.propogate("edge", data)
            g = self.propogate("glob", data)
        return GraphBatch(
            x, e, g, edges=data.edges, node_idx=data.node_idx, edge_idx=data.edge_idx
        )


class NNGraphNetwork(NNGraph):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.core = Core(config)
        self.decoder = Encoder(config)
        self.out = OutTransform(config)

    def forward(self, data, steps, save_all=True):
        latent0 = self.encoder(data)
        data = self.encoder(data)
        out_arr = []
        for _ in range(steps):
            data = GraphBatch.cat(latent0, data)
            data = self.core(data)
            data = self.decoder(data)
            out = self.out(data)
            out = GraphTuple(x=out.x, e=out.e, g=out.g)
            if save_all:
                out_arr.append(out)
            else:
                out_arr = [out]
        return out_arr

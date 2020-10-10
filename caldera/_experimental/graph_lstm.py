from .nngraph import NNGraph
from caldera.data import GraphBatch
from torch import nn

class GraphEncoder(NNGraph):

    def __init__(self, node, edge, glob):
        super().__init__()

        self.add_node(node, 'node')
        self.add_node(edge, 'edge')
        self.add_node(glob, 'glob')

        self.add_edge(lambda data: data.x, 'node')
        self.add_edge(lambda data: data.e, 'edge')
        self.add_edge(lambda data: data.g, 'glob')

    def forward(self, data: GraphBatch):
        with self.run():
            x = self.propogate('node', data)
            e = self.propogate('edge', data)
            g = self.propogate('glob', data)
        return data.new_like(x, e, g)


# class GraphLSTM(nn.Module):
#
#     def __init__(self):
#         self.
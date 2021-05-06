from .nngraph import NNGraph
from caldera.data import GraphBatch
from torch import nn
from caldera import gnn


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


class GraphSigmoid(GraphEncoder):

    def __init__(self):
        super().__init__(
            nn.Sequential(
                gnn.Flex(nn.Linear)(..., gnn.Flex.d(0)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                gnn.Flex(nn.Linear)(..., gnn.Flex.d(0)),
                nn.Sigmoid()
            ),
            nn.Sequential(
                gnn.Flex(nn.Linear)(..., gnn.Flex.d(0)),
                nn.Sigmoid()
            ),
        )


class GraphLSTM(nn.Module):

    def __init__(self):
        self.sig1 = GraphSigmoid()
        self.sig2 = GraphSigmoid()
        self.sig3 = GraphSigmoid()
        self.tanh = nn.Tanh()

    def forward(self, data):

        state = data[0]
        prev_data = data[0]

        for _data in data:
            # forget
            state = state * self.sig1(prev_data.cat(data))

            # remember
            state = self.sig2(data) * self.tanh(data) + state

            # out
            data = self.tanh(state) * self.sig3(data)

        return state, data


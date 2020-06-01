import torch

from pyrographnets.data import GraphData
from pyrographnets.utils import scatter_group
from typing import List
import networkx as nx


class GraphBatch(GraphData):
    __slots__ = GraphData.__slots__ + ['node_idx', 'edge_idx']

    def __init__(self, node_attr, edge_attr, global_attr, edges, node_idx, edge_idx):
        super(GraphBatch, self).__init__(node_attr, edge_attr, global_attr, edges)
        self.node_idx = node_idx
        self.edge_idx = edge_idx
        GraphBatch.debug(self)

    @staticmethod
    def _same(a):
        return min(a) == max(a)

    def debug(self):
        super().debug()
        if not self.node_idx.dtype == torch.long:
            raise RuntimeError(
                "Wrong tensor type. `node_idx` must be dtype={} not {}".format(self.node_idx.dtype, torch.long))
        if not self.edge_idx.dtype == torch.long:
            raise RuntimeError(
                "Wrong tensor type. `edge_idx` must be dtype={} not {}".format(self.edge_idx.dtype, torch.long))
        if not self.node_idx.max() == self.edge_idx.max():
            raise RuntimeError("Number of graphs in node_idx and edge_idx mismatch")
        if not self.node_idx.min() == 0:
            raise RuntimeError(
                "Minimum graph index (node_idx.min()) must start at 0, not {}".format(self.node_idx.min()))
        if not self.edge_idx.min() == 0:
            raise RuntimeError(
                "Minimum graph index (edge_idx.min()) must start at 0, not {}".format(self.edge_idx.min()))

    @classmethod
    def from_data_list(cls, data_list):
        # checks
        n_features = []
        e_features = []
        g_features = []
        for data in data_list:
            n_features.append(data.x.shape[1])
            e_features.append(data.e.shape[1])
            g_features.append(data.g.shape[0])
        if not cls._same(n_features):
            raise RuntimeError("Node feature dimensions must all be the same")
        if not cls._same(e_features):
            raise RuntimeError("Edge feature dimensions must all be the same")
        if not cls._same(g_features):
            raise RuntimeError("Global feature dimensions must all be the same")

        node_repeats = torch.tensor([data.x.shape[0] for data in data_list])
        edge_repeats = torch.tensor([data.e.shape[0] for data in data_list])
        node_idx = torch.repeat_interleave(torch.range(0, node_repeats.shape[0] - 1, dtype=torch.long), node_repeats)
        edge_idx = torch.repeat_interleave(torch.range(0, edge_repeats.shape[0] - 1, dtype=torch.long), edge_repeats)

        # concatenate edges
        edges = torch.cat([data.edges for data in data_list], dim=1)

        # cumulated shapes
        c = torch.cumsum(torch.tensor([0] + [data.x.shape[0] for data in data_list[:-1]]), dim=0)
        delta = torch.repeat_interleave(c, edge_repeats).repeat(2, 1)

        # shift concatenated edges
        edges = edges + delta

        return cls(
            node_attr=torch.cat([data.x for data in data_list]),
            edge_attr=torch.cat([data.e for data in data_list]),
            global_attr=torch.cat([data.g for data in data_list]),
            edges=edges,
            node_idx=node_idx,
            edge_idx=edge_idx
        )

    def to_data_list(self) -> List[GraphData]:
        gidx_n, node_attr = scatter_group(self.x, self.node_idx)
        gidx_e, edge_attr = scatter_group(self.e, self.edge_idx)
        gidx_edge, edges = scatter_group(self.edges.T, self.edge_idx)

        def to_dict(a, b):
            return dict(zip([x.item() for x in a], b))

        ndict = to_dict(gidx_n, node_attr)
        edict = to_dict(gidx_e, edge_attr)
        edgesdict = to_dict(gidx_edge, edges)
        datalist = []
        for k in ndict:
            _edges = edgesdict[k].T - edgesdict[k].min()

            data = GraphData(ndict[k], edict[k], self.g[k], _edges)
            datalist.append(data)
        return datalist

    def to_networkx(self, *args, **kwargs):
        raise NotImplementedError

    def from_networkx(self, *args, **kwargs):
        raise NotImplementedError

    def to_networkx_list(self, feature_key: str = 'features', global_attr_key: str = 'data') -> List[nx.DiGraph]:
        graphs = []
        for data in self.to_data_list():
            graphs.append(data.to_networkx(feature_key, global_attr_key=global_attr_key))

    @staticmethod
    def from_networkx_list(graphs: List[nx.DiGraph], *args, **kwargs) -> 'GraphBatch':
        data_list = [GraphData.from_networkx(g, *args, **kwargs) for g in graphs]
        return GraphBatch.from_data_list(data_list)
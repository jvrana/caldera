from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List
import torch


class GraphData(object):
    """Data representing a single graph"""
    __slots__ = ['x', 'e', 'g', 'edges']

    def __init__(self, node_attr, edge_attr, global_attr, edges):
        self.x = node_attr
        self.e = edge_attr
        self.g = global_attr
        self.edges = edges
        GraphData.debug(self)

    def debug(self):
        if self.edges.max() >= self.x.shape[0]:
            raise RuntimeError(
                "Edge coordinate {} is greater than number of nodes {}".format(self.edges.max(), self.x.shape[0
                ]))
        if not self.edges.shape[1] == self.e.shape[0]:
            raise RuntimeError("Number of edges {} must match number of edge attributes {}".format(
                self.edges.shape[1],
                self.e.shape[0]
            ))

        if not self.edges.dtype == torch.long:
            raise RuntimeError(
                "Wrong tensor type. `edges` must be dtype={} not {}".format(self.edges.dtype, torch.long))

    def apply(self, func):
        for f in self.__slots__:
            func(f)

    def to(self, device: str):
        self.apply(lambda x: x.to(device))

    def contiguous(self):
        self.apply(lambda x: x.contiguous())

    @property
    def num_graphs(self):
        return self.g.shape[0]

    @property
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def node_shape(self):
        return self.x.shape[1:]

    @property
    def edge_shape(self):
        return self.e.shape[1:]

    @property
    def global_shape(self):
        return self.g.shape[1:]

    @property
    def shape(self):
        return self.x.shape[1:] + self.e.shape[1:] + self.g.shape[1:]

    @property
    def size(self):
        return self.x.shape[:1] + self.e.shape[:1] + self.g.shape[:1]

    def _mask_fields(self, masks: Dict[str, torch.tensor]):
        for m in masks:
            if m not in self.__slots__:
                raise RuntimeError("{} is not a valid field".format(m))
        masked_fields = []
        for field in self.__slots__:
            if field not in masks or masks[field] is None:
                masked_fields.append(getattr(self, field))
            else:
                masked_fields.append(getattr(self, field)[:, masks[field]])
        return masked_fields

    def mask(self, node_mask, edge_mask, global_mask, invert: bool = False):
        d = {'x': node_mask, 'e': edge_mask, 'g': global_mask}
        if invert:
            d = {k: ~v for k, v in d.items()}
        return self.__class__(
            *self._mask_fields(d)
        )

    def clone(self):
        return self.__class__(
            *[getattr(self, field).clone() for field in self.__class__.__slots__]
        )

    def __repr__(self):
        return "<{cls} size(n,e,g)={size} features(n,e,g)={shape}>".format(
            cls=self.__class__.__name__,
            n_graphs=self.num_graphs,
            size=self.x.shape[:1] + self.e.shape[:1] + self.g.shape[:1],
            shape=self.shape
        )


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


def random_data(n_features, e_features, g_features):
    n_nodes = torch.randint(1, 10, torch.Size([])).item()
    n_edges = torch.randint(1, 20, torch.Size([])).item()
    return GraphData(
        torch.randn(n_nodes, n_features),
        torch.randn(n_edges, e_features),
        torch.randn(1, g_features),
        torch.randint(0, n_nodes, torch.Size([2, n_edges]))
    )


from torch.utils.data import DataLoader


def collate(data_list):
    if isinstance(data_list[0], tuple):
        if issubclass(type(data_list[0][0]), GraphData):
            return tuple([collate([x[i] for x in data_list]) for i in range(len(data_list[0]))])
        else:
            raise RuntimeError(
                "Cannot collate {}({})({})".format(type(data_list), type(data_list[0]), type(data_list[0][0])))
    return GraphBatch.from_data_list(data_list)


class GraphDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 **kwargs):
        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=collate, **kwargs)
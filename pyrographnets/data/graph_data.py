from typing import Dict

import torch
from pyrographnets.utils import _first
from typing import Optional
import numpy as np
import networkx as nx
from typing import Callable

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

        if not self.edges.shape[0] == 2:
            raise RuntimeError(
                "Edges must be a tensor of shape `[2, num_edges]`"
            )

        if not self.x.ndim == 2:
            raise RuntimeError(
                "Node attr must have 2 dimensions"
            )

        if not self.g.ndim == 2:
            raise RuntimeError(
                "Global attr must have 2 dimensions"
            )

        if not self.e.ndim == 2:
            raise RuntimeError(
                "Edge attr must have 2 dimensions"
            )

        if not self.edges.ndim == 2:
            raise RuntimeError(
                "Edges must have 2 dimensions"
            )

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

    @staticmethod
    def from_networkx(g: nx.DiGraph,
                      n_node_feat: Optional[int] = None,
                      n_edge_feat: Optional[int] = None,
                      n_glob_feat: Optional[int] = None,
                      feature_key: str = 'features',
                      global_attr_key: str = 'data'):
        if hasattr(g, global_attr_key):
            gdata = getattr(g, global_attr_key)
        else:
            gdata = {}

        if n_node_feat is None:
            _, ndata = _first(g.nodes(data=True))
            n_node_feat = ndata[feature_key].shape[0]

        if n_edge_feat is None:
            _, _, edata = _first(g.edges(data=True))
            n_edge_feat = edata[feature_key].shape[0]

        if n_glob_feat is None:
            n_glob_feat = gdata[feature_key].shape[0]

        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        node_attr = np.empty((n_nodes, n_node_feat))
        edge_attr = np.empty((n_edges, n_edge_feat))
        glob_attr = np.empty((1, n_glob_feat))

        nodes = sorted(list(g.nodes(data=True)))
        ndict = {}
        for i, (n, ndata) in enumerate(nodes):
            node_attr[i] = ndata[feature_key]
            ndict[n] = i

        edges = np.empty((2, n_edges))
        for i, (n1, n2, edata) in enumerate(g.edges(data=True)):
            edges[:, i] = np.array([ndict[n1], ndict[n2]])
            edge_attr[i] = edata[feature_key]

        glob_attr[0] = getattr(g, global_attr_key)[feature_key]

        return GraphData(torch.tensor(node_attr, dtype=torch.float),
                         torch.tensor(edge_attr, dtype=torch.float),
                         torch.tensor(glob_attr, dtype=torch.float),
                         torch.tensor(edges, dtype=torch.long))

    def to_networkx(self, feature_key: str = 'features',
                    global_attr_key: str = 'data'):
        g = nx.DiGraph()
        for n, ndata in enumerate(self.x):
            g.add_node(n, **{feature_key: ndata})
        for i, e in enumerate(self.edges.T):
            g.add_edge(e[0].item(), e[1].item(), **{feature_key: self.e[i]})
        setattr(g, global_attr_key, {feature_key: self.g.clone()})
        return g

    def __repr__(self):
        return "<{cls} size(n,e,g)={size} features(n,e,g)={shape}>".format(
            cls=self.__class__.__name__,
            n_graphs=self.num_graphs,
            size=self.x.shape[:1] + self.e.shape[:1] + self.g.shape[:1],
            shape=self.shape
        )

    def _eq_helper(self, other: 'GraphData', comparator: Callable[[torch.Tensor, torch.Tensor], bool]) -> bool:
        if not torch.all(torch.eq(self.edges, other.edges)):
            return False
        for attr in ['x', 'e', 'g']:
            a = getattr(self, attr)
            b = getattr(other, attr)
            if not comparator(a, b):
                return False
        return True

    def __eq__(self, other: 'GraphData') -> bool:
        def is_eq(a, b):
            if a.shape != b.shape:
                return False
            return torch.all(torch.eq(a, b))
        return self._eq_helper(other, comparator=is_eq)

    def allclose(self, other: 'GraphData', **kwargs) -> bool:
        def _allclose(a, b):
            return torch.allclose(a, b, **kwargs)
        return self._eq_helper(other, comparator=_allclose)
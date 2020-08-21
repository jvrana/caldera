from __future__ import annotations

from typing import List
from typing import Optional
from typing import Type

import networkx as nx
import torch

from caldera.data.graph_data import GraphData
from caldera.utils.nx_utils import DirectedGraph
from caldera.utils import scatter_group
from caldera.utils import stable_arg_sort_long
from caldera.utils import reindex_tensor


class GraphBatch(GraphData):
    __slots__ = GraphData.__slots__ + ["node_idx", "edge_idx"]

    # TODO: global_idx
    def __init__(
        self,
        node_attr: torch.FloatTensor,
        edge_attr: torch.FloatTensor,
        global_attr: torch.FloatTensor,
        edges: torch.LongTensor,
        node_idx: torch.LongTensor,
        edge_idx: torch.LongTensor,
    ):
        super().__init__(node_attr, edge_attr, global_attr, edges)
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
                "Wrong tensor type. `node_idx` must be dtype={} not {}".format(
                    self.node_idx.dtype, torch.long
                )
            )
        if not self.edge_idx.dtype == torch.long:
            raise RuntimeError(
                "Wrong tensor type. `edge_idx` must be dtype={} not {}".format(
                    self.edge_idx.dtype, torch.long
                )
            )
        if (
            self.node_idx.shape[0]
            and self.edge_idx.shape[0]
            and self.node_idx.max() != self.edge_idx.max()
        ):
            raise RuntimeError(
                "Number of graphs in node_idx {} and edge_idx {} mismatch".format(
                    self.node_idx.max(), self.edge_idx.max()
                )
            )
        if self.node_idx.shape[0] and self.node_idx.shape[0] != self.x.shape[0]:
            raise RuntimeError(
                "Number of node indices {} must match number of node attr {}".format(
                    self.node_idx.shape[0], self.x.shape[0]
                )
            )
        if self.edge_idx.shape[0] and self.edge_idx.shape[0] != self.e.shape[0]:
            raise RuntimeError(
                "Number of node indices {} must match number of node attr {}".format(
                    self.edge_idx.shape[0], self.e.shape[0]
                )
            )
        # if not self.node_idx.min() == 0:
        #     raise RuntimeError(
        #         "Minimum graph index (node_idx.min()) must start at 0, not {}".format(self.node_idx.min()))
        # if not self.edge_idx.min() == 0:
        #     raise RuntimeError(
        #         "Minimum graph index (edge_idx.min()) must start at 0, not {}".format(self.edge_idx.min()))

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
        node_idx = torch.repeat_interleave(
            torch.arange(0, node_repeats.shape[0], dtype=torch.long), node_repeats
        )
        edge_idx = torch.repeat_interleave(
            torch.arange(0, edge_repeats.shape[0], dtype=torch.long), edge_repeats
        )

        # concatenate edges
        edges = torch.cat([data.edges for data in data_list], dim=1)

        # cumulated shapes
        c = torch.cumsum(
            torch.tensor([0] + [data.x.shape[0] for data in data_list[:-1]]), dim=0
        )
        delta = torch.repeat_interleave(c, edge_repeats).repeat(2, 1)

        # shift concatenated edges
        edges = edges + delta

        return cls(
            node_attr=torch.cat([data.x for data in data_list]),
            edge_attr=torch.cat([data.e for data in data_list]),
            global_attr=torch.cat([data.g for data in data_list]),
            edges=edges,
            node_idx=node_idx,
            edge_idx=edge_idx,
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
        n_sizes = 0
        for k in ndict:
            _edges = edgesdict[k].T
            _edges -= n_sizes
            n_sizes += ndict[k].shape[0]

            data = GraphData(ndict[k], edict[k], torch.unsqueeze(self.g[k], 0), _edges)
            datalist.append(data)
        return datalist

    def to_networkx(self, *args, **kwargs):
        raise NotImplementedError

    def from_networkx(self, *args, **kwargs):
        raise NotImplementedError

    def to_networkx_list(
        self,
        feature_key: str = "features",
        global_attr_key: str = "data",
        graph_type: Type[DirectedGraph] = nx.OrderedMultiDiGraph,
    ) -> List[DirectedGraph]:
        graphs = []
        for data in self.to_data_list():
            graphs.append(
                data.to_networkx(
                    feature_key, global_attr_key=global_attr_key, graph_type=graph_type
                )
            )
        return graphs

    @staticmethod
    def from_networkx_list(
        graphs: List[DirectedGraph], *args, **kwargs
    ) -> "GraphBatch":
        data_list = [GraphData.from_networkx(g, *args, **kwargs) for g in graphs]
        return GraphBatch.from_data_list(data_list)

    def append_nodes(self, node_attr: torch.Tensor, node_idx: torch.Tensor):
        datalist = self.to_data_list()
        idx, groups = scatter_group(node_attr, node_idx)
        for i, g in zip(idx, groups):
            data = datalist[i.item()]
            data.append_nodes(g)
        batch = GraphBatch.from_data_list(datalist)
        self.x = batch.x
        self.e = batch.e
        self.g = batch.g
        self.node_idx = batch.node_idx
        self.debug()
        return self

    def append_edges(
        self, edge_attr: torch.Tensor, edges: torch.Tensor, edge_idx: torch.Tensor
    ):
        """Append edges to the graph batch at the specified edge_idx (assumed
        to be sorted).

        :param edge_attr:
        :param edges:
        :param edge_idx:
        :return:
        """
        edges = torch.cat([self.edges, edges], dim=1)
        e = torch.cat([self.e, edge_attr])
        edge_idx = torch.cat([self.edge_idx, edge_idx])
        i = stable_arg_sort_long(edge_idx)

        self.edges = edges[:, i]
        self.e = e[i]
        self.edge_idx = edge_idx[i]
        self.debug()
        return self

    def _mask_dispatch(
        self,
        node_mask: Optional[torch.BoolTensor],
        edge_mask: Optional[torch.BoolTensor],
        as_view: bool,
        detach: bool,
        new_inst: bool,
    ):
        self._validate_masks(node_mask, edge_mask)
        edges = self._apply_mask(self.edges, edge_mask, detach, as_view, dim=1)
        edges = self._mask_dispatch_reindex_edges(edges, node_mask)
        x = self._apply_mask(self.x, node_mask, detach, as_view)
        e = self._apply_mask(self.e, edge_mask, detach, as_view)
        g = self._apply_mask(self.g, None, detach, as_view)
        node_idx = self._apply_mask(self.node_idx, node_mask, detach, as_view)
        edge_idx = self._apply_mask(self.edge_idx, edge_mask, detach, as_view)

        node_idx, edge_idx = reindex_tensor(node_idx, edge_idx)

        return self._mask_dispatch_constructor(
            new_inst, x, e, g, edges, node_idx=node_idx, edge_idx=edge_idx
        )

    # def append_edges
    # def collect_and_collate(x1, i1, x2, i2, collate_fn = torch.cat):
    #     i1, groups1 = scatter_group(x1, i1)
    #     i2, groups2 = scatter_group(x2, i2)
    #     d1 = {k.item(): v for k, v in zip(i1, groups1)}
    #     d2 = {k.item(): v for k, v in zip(i2, groups2)}
    #     return dict_collate(d1, d2, torch.cat)
    #
    #
    # i = stable_arg_sort_long(edge_idx)
    # edge_attr = edge_attr[i]
    # edges = edges[:, i]
    #
    # d = collect_and_collate(edge_attr, edge_idx, self.e, self.edge_idx)
    # keys = sorted(d)
    # e = torch.cat([d[k] for k in keys])
    #
    # d = collect_and_collate(edges.T, edge_idx, self.edges.T, self.edge_idx)
    # keys = sorted(d)
    # edges = torch.cat([d[k] for k in keys], dim=0)
    #
    # d = collect_and_collate(edge_idx, edge_idx, self.edge_idx, self.edge_idx)
    # keys = sorted(d)
    # edge_idx = torch.cat([d[k] for k in keys], dim=0)
    #
    # self.e = e
    # self.edges = edges.T
    # self.edge_idx = edge_idx
    # self.debug()
    # return self
    # e = torch.cat([d[k] for k in keys])

    #
    # def collate_shape(x):
    #     return [[i] * _x.shape[0] for i, _x in enumerate(x)]
    #
    # d = dict_collate(d1, d2, collate_shape)
    # u = [torch.Tensor(functools.reduce(operator.add, v)) for v in d.values()]
    # edge_idx = torch.cumsum(torch.cat(u), 0)
    #
    # self.e = e
    # self.edge_idx = edge_idx
    # self.debug()
    # return self

    # i1, g1 = scatter_group(self.x, self.node_idx)
    # i2, g2 = scatter_group(node_attr, node_idx)
    # d1 = {k.item(): v for k, v in zip(i1, g1)}
    # d2 = {k.item(): v for k, v in zip(i2, g2)}
    #
    # # collate
    # d = dict_collate(d1, d2, torch.cat)
    # keys = sorted(d.keys())
    # x = torch.cat([d[k] for k in keys])
    #
    # # collect node indices
    # node_idx = []
    # for k in keys:
    #     node_idx += [k] * d[k].shape[0]
    #
    # # correct edges
    # def get_shape(x):
    #     return [[i] * _x.shape[0] for i, _x in enumerate(x)]
    #
    # u = [torch.Tensor(functools.reduce(operator.add, v)) for v in dict_collate(d1, d2, get_shape).values()]
    # delta_edges = torch.cumsum(torch.cat(u), 0)
    # self.edges += delta_edges
    # self.x = x
    # self.node_idx = torch.tensor(node_idx, dtype=torch.long)
    # self.debug()
    #
    # def append_nodes(self, edge_attr: torch.Tensor, edge_idx: torch.Tensor):
    #     i1, g1 = scatter_group(self.x, self.node_idx)
    #     i2, g2 = scatter_group(edge_attr, node_idx)
    #     d1 = {k.item(): v for k, v in zip(i1, g1)}
    #     d2 = {k.item(): v for k, v in zip(i2, g2)}
    #
    #     # collate
    #     d = dict_collate(d1, d2, torch.cat)
    #     keys = sorted(d.keys())
    #     x = torch.cat([d[k] for k in keys])
    #
    #     # collect node indices
    #     node_idx = []
    #     for k in keys:
    #         node_idx += [k] * d[k].shape[0]
    #
    #     self.x = x
    #     self.node_idx = torch.tensor(node_idx, dtype=torch.long)
    #     self.debug()

    def _eq_helper(self, *args, **kwargs):
        raise NotImplementedError("Cannot compare batches")

    def allclose(self, *args, **kwargs):
        raise NotImplementedError("Cannot compare batches")

    @classmethod
    def random(cls, *args, **kwargs):
        raise NotImplementedError("Not implemented for {}".format(cls))

    @classmethod
    def random_batch(
        cls,
        size: int,
        n_feat: int,
        e_feat: int,
        g_feat: int,
        min_nodes: int = 1,
        max_nodes: int = 10,
        min_edges: int = 1,
        max_edges: int = 20,
        requires_grad: Optional[bool] = None,
    ) -> GraphBatch:
        datalist = []
        for _ in range(size):
            datalist.append(
                GraphData.random(
                    n_feat,
                    e_feat,
                    g_feat,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
                    min_edges=min_edges,
                    max_edges=max_edges,
                )
            )
        batch = cls.from_data_list(datalist)
        if requires_grad is not None:
            batch.requires_grad = requires_grad
        return batch

    def __eq__(self, *args, **kwargs):
        raise NotImplementedError("Cannot compare batches")

    def view(
        self,
        x_slice: Optional[slice] = None,
        e_slice: Optional[slice] = None,
        g_slice: Optional[slice] = None,
        edges_slice: Optional[slice] = None,
    ) -> GraphData:
        if x_slice is None:
            x_slice = slice(None, None, None)
        if e_slice is None:
            e_slice = slice(None, None, None)
        if g_slice is None:
            g_slice = slice(None, None, None)
        if edges_slice is None:
            edges_slice = slice(None, None, None)
        return self.__class__(
            self.x[:, x_slice],
            self.e[:, e_slice],
            self.g[:, g_slice],
            self.edges[:, edges_slice],
            self.node_idx[:],
            self.edge_idx[e_slice],
        )

    def disjoint_union(self, other: GraphBatch) -> GraphBatch:
        """
        Disjoint union between two GraphBatches.

        :param other:
        :return:
        """
        x = torch.cat(self.x, other.x)
        e = torch.cat(self.e, other.e)
        g = torch.cat(self.g, other.g)

        n = self.node_idx.max()
        node_idx = torch.cat(self.node_idx, other.node_idx + n)
        edge_idx = torch.cat(self.edge_idx, other.edge_idx + n)
        edges = torch.cat(self.edges, other.edges + n)
        node_idx, edge_idx, edges = reindex_tensor(node_idx, edge_idx, edges)

        return GraphBatch(
            x, e, g, edges, node_idx, edge_idx
        )

    def shuffle_graphs_(self) -> None:
        b = torch.unique(self.node_idx)
        ridx = torch.randperm(b.shape[0])
        self.g = self.g[ridx]
        _, node_idx, edge_idx = reindex_tensor(ridx, self.node_idx, self.edge_idx)
        self.node_idx = node_idx
        self.edge_idx = edge_idx

    def shuffle_graphs(self) -> GraphBatch:
        cloned = self.clone()
        cloned.shuffle_graphs_()
        return cloned

    def shuffle_(self) -> None:
        self.shuffle_graphs_()
        self.shuffle_nodes_()
        self.shuffle_edges_()

    def shuffle(self) -> GraphBatch:
        cloned = self.clone()
        cloned.shuffle_()
        return cloned
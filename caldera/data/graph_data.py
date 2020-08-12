from __future__ import annotations

import functools
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import networkx as nx
import numpy as np
import torch

from caldera.utils import _first
from caldera.utils import same_storage

GraphType = TypeVar("GraphType", nx.MultiDiGraph, nx.OrderedMultiDiGraph, nx.DiGraph)


# TODO: there should be a super class, TorchComposition, with apply methods etc.
# TODO: support n dim tensors
# TODO: implicit support for torch.Tensor
# TODO: handle empty features and targets
class GraphData:
    """Data representing a single graph."""

    __slots__ = ["x", "e", "g", "edges"]
    _differentiable = ["x", "e", "g"]

    def __init__(
        self,
        node_attr,
        edge_attr,
        global_attr,
        edges,
        requires_grad: Optional[bool] = None,
    ):
        self.x = node_attr
        self.e = edge_attr
        self.g = global_attr
        self.edges = edges
        GraphData.debug(self)
        if requires_grad is not None:
            self.requires_grad = requires_grad

    def debug(self):
        if (
            self.edges.shape[0]
            and self.edges.shape[1]
            and self.edges.max() >= self.x.shape[0]
        ):
            raise RuntimeError(
                "Edge coordinate {} is greater than number of nodes {}".format(
                    self.edges.max(), self.x.shape[0]
                )
            )
        if not self.edges.shape[1] == self.e.shape[0]:
            raise RuntimeError(
                "Number of edges {} must match number of edge attributes {}".format(
                    self.edges.shape[1], self.e.shape[0]
                )
            )

        if not self.edges.dtype == torch.long:
            raise RuntimeError(
                "Wrong tensor type. `edges` must be dtype={} not {}".format(
                    torch.long, self.edges.dtype
                )
            )

        if not self.edges.shape[0] == 2:
            raise RuntimeError("Edges must be a tensor of shape `[2, num_edges]`")

        if not self.x.ndim == 2:
            raise RuntimeError("Node attr must have 2 dimensions")

        if not self.g.ndim == 2:
            raise RuntimeError("Global attr must have 2 dimensions")

        if not self.e.ndim == 2:
            raise RuntimeError("Edge attr must have 2 dimensions")

        if not self.edges.ndim == 2:
            raise RuntimeError("Edges must have 2 dimensions")

    def _apply(
        self,
        func,
        new_inst: bool,
        args: Tuple[Any, ...] = tuple(),
        kwargs: Dict = None,
        keys: Optional[Tuple[str]] = None,
    ) -> "GraphData":
        """Applies the function to the graph. Be mindful of what the function
        is doing.

        Ask the following before using this function:

        1. Is it applying the function in place? copying it?
        2. Is the function recorded in the computation graph? Will you need to detach it?

        :param func: function to apply
        :param new_inst: Whether to create a new GraphData instance. Values will be copied (and detached by default).
        :param detatch_new_inst: Detaches copied vectors from the computation graph. This should almost always be left
            `True`.
        :param args:
        :param kwargs:
        :return:
        """
        if keys is None:
            keys = self.__slots__
        if kwargs is None:
            kwargs = {}
        init_args = []
        for f in keys:
            old_val = getattr(self, f)
            val = func(old_val, *args, **kwargs)
            if new_inst:
                init_args.append(val)
        if new_inst:
            return self.__class__(*init_args)
        return self

    # TODO: finish clone, copy, apply, etc.
    def apply(self, func, *args, keys: Optional[Tuple[str]] = None, **kwargs):
        """Applies the function to the data, creating a new instance of
        GraphData."""
        return self._apply(func, new_inst=True, args=args, kwargs=kwargs, keys=keys)

    def apply_(self, func, *args, keys: Optional[Tuple[str]] = None, **kwargs):
        """Applies the function in place to the data, wihout creating a new
        instance of GraphData."""
        return self._apply(func, new_inst=False, args=args, kwargs=kwargs, keys=keys)

    def to(self, device: str, *args, **kwargs):
        return self.apply(lambda x: x.to(device, *args, **kwargs))

    def share_storage(
        self, other: GraphData, return_dict: Optional[bool] = False
    ) -> Union[Dict[str, bool], bool]:
        """Check if this data shares storage with another data.

        :param other: The other GraphData object.
        :param return_dict: if true, return dictionary of which tensors share the same storage. Else returns true if
            any tensors share the same storage.
        :return:
        """
        d = {}
        for k in self.__slots__:
            a = getattr(self, k)
            b = getattr(other, k)
            c = same_storage(a, b)
            if return_dict:
                d[k] = c
            elif c:
                return True
        return d

    def contiguous(self):
        return self.apply(lambda x: x.contiguous())

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
        d = {"x": node_mask, "e": edge_mask, "g": global_mask}
        if invert:
            d = {k: ~v for k, v in d.items()}
        return self.__class__(*self._mask_fields(d))

    @property
    def requires_grad(self):
        g = [getattr(self, k).requires_grad for k in self._differentiable]
        return all(g)

    @requires_grad.setter
    def requires_grad(self, v):
        def set_requires_grad(x):
            x.requires_grad = v

        self.apply_(set_requires_grad, keys=self._differentiable)

    # TODO: clone tests
    def clone(self):
        """Clones the data.

        Note that like the `clone()` method, this function will be
        recorded in the computation graph.
        """
        return self.apply(lambda x: x.clone())

    # TODO: copy tests
    def copy(self, non_blocking: bool = False, *emtpy_like_args, **emtpy_like_kwargs):

        """non_blocking (bool) – if True and this copy is between CPU and GPU,
        the copy may occur asynchronously with respect to the host. For other
        cases, this argument has no effect.

        :return:
        """
        """Unlike clone, copies the data *without the computation graph*"""
        return self.apply(
            lambda x: torch.empty_like(x, *emtpy_like_args, **emtpy_like_kwargs).copy_(
                x, non_blocking=non_blocking
            )
        )

    # TODO: docstrings
    # TODO: handle undirected and hypergraphs
    # TODO: check that features are NUMPY rather than TORCH
    @classmethod
    def from_networkx(
        cls,
        g: GraphType,
        n_node_feat: Optional[int] = None,
        n_edge_feat: Optional[int] = None,
        n_glob_feat: Optional[int] = None,
        feature_key: str = "features",
        global_attr_key: str = "data",
        requires_grad: Optional[bool] = None,
        dtype: str = torch.float32,
    ):
        """

        :param g:
        :param n_node_feat:
        :param n_edge_feat:
        :param n_glob_feat:
        :param feature_key: The key to look for data in node, edge, and global data.
        :param global_attr_key: Key to look for global data.
        :return:
        """
        if hasattr(g, global_attr_key):
            gdata = getattr(g, global_attr_key)
        else:
            gdata = {}

        if n_node_feat is None:
            try:
                _, ndata = _first(g.nodes(data=True))
                if feature_key not in ndata:
                    n_node_feat = 0
                else:
                    n_node_feat = ndata[feature_key].size
            except StopIteration:
                n_node_feat = 0

        if n_edge_feat is None:
            try:
                _, _, edata = _first(g.edges(data=True))
                if feature_key not in edata:
                    n_edge_feat = 0
                else:
                    n_edge_feat = edata[feature_key].size
            except StopIteration:
                n_edge_feat = 0

        if n_glob_feat is None:
            if feature_key in gdata:
                if feature_key not in gdata:
                    n_glob_feat = 0
                else:
                    n_glob_feat = gdata[feature_key].size
            else:
                n_glob_feat = 0

        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        node_attr = np.empty((n_nodes, n_node_feat))
        edge_attr = np.empty((n_edges, n_edge_feat))
        glob_attr = np.empty((1, n_glob_feat))

        nodes = sorted(list(g.nodes(data=True)))
        ndict = {}
        # TODO: support n dim tensors (but they get flattened anyways...)
        for i, (n, ndata) in enumerate(nodes):
            node_attr[i] = ndata[feature_key].flatten()
            ndict[n] = i

        # TODO: method to change dtype?
        edges = np.empty((2, n_edges), dtype=np.float)
        for i, (n1, n2, edata) in enumerate(g.edges(data=True)):
            edges[:, i] = np.array([ndict[n1], ndict[n2]])
            edge_attr[i] = edata[feature_key].flatten()

        if feature_key in gdata:
            glob_attr[0] = gdata[feature_key]

        if requires_grad is not None:
            tensor = functools.partial(
                torch.tensor, dtype=dtype, requires_grad=requires_grad
            )
        else:
            tensor = functools.partial(torch.tensor, dtype=dtype)

        data = GraphData(
            tensor(node_attr),
            tensor(edge_attr),
            tensor(glob_attr),
            torch.tensor(edges, dtype=torch.long),
        )
        return data

    def to_networkx(
        self,
        feature_key: str = "features",
        global_attr_key: str = "data",
        graph_type: Type[GraphType] = nx.OrderedMultiDiGraph,
    ) -> GraphType:
        g = graph_type()
        for n, ndata in enumerate(self.x):
            g.add_node(n, **{feature_key: ndata})
        g.ordered_edges = []
        for i, e in enumerate(self.edges.T):
            n = g.add_edge(e[0].item(), e[1].item(), **{feature_key: self.e[i]})
            g.ordered_edges.append((e[0].item(), e[1].item(), n))
        setattr(g, global_attr_key, {feature_key: self.g.clone()})
        return g

    def __repr__(self):
        return "<{cls} size(n,e,g)={size} features(n,e,g)={shape}>".format(
            cls=self.__class__.__name__,
            n_graphs=self.num_graphs,
            size=self.x.shape[:1] + self.e.shape[:1] + self.g.shape[:1],
            shape=self.shape,
        )

    def _eq_helper(
        self,
        other: "GraphData",
        comparator: Callable[[torch.Tensor, torch.Tensor], bool],
    ) -> bool:
        if not torch.all(torch.eq(self.edges, other.edges)):
            return False
        for attr in ["x", "e", "g"]:
            a = getattr(self, attr)
            b = getattr(other, attr)
            if not comparator(a, b):
                return False
        return True

    def __eq__(self, other: "GraphData") -> bool:
        def is_eq(a, b):
            if a.shape != b.shape:
                return False
            return torch.all(torch.eq(a, b))

        return self._eq_helper(other, comparator=is_eq)

    # TODO: implement this for batch?
    def append_nodes(self, node_attr: torch.Tensor):
        assert isinstance(self, GraphData)
        if not node_attr.ndim == 2:
            raise RuntimeError("Node attributes must have 2 dimensions")

        self.x = torch.cat([self.x, node_attr])
        self.debug()

    def append_edges(self, edge_attr: torch.Tensor, edges: torch.Tensor):
        assert isinstance(self, GraphData)
        self.edges = torch.cat([self.edges, edges], dim=1)
        self.e = torch.cat([self.e, edge_attr])
        self.debug()

    def allclose(self, other: "GraphData", **kwargs) -> bool:
        def _allclose(a, b):
            return torch.allclose(a, b, **kwargs)

        return self._eq_helper(other, comparator=_allclose)

    @classmethod
    def random(
        cls, n_feat: int, e_feat: int, g_feat: int, requires_grad: Optional[bool] = None
    ) -> GraphData:
        n_nodes = torch.randint(1, 10, torch.Size([])).item()
        n_edges = torch.randint(1, 20, torch.Size([])).item()
        return cls(
            torch.randn(n_nodes, n_feat),
            torch.randn(n_edges, e_feat),
            torch.randn(1, g_feat),
            torch.randint(0, n_nodes, torch.Size([2, n_edges])),
            requires_grad=requires_grad,
        )

    # TODO: view
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
        )

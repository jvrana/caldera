from __future__ import annotations

import functools
from functools import reduce
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import networkx as nx
import numpy as np
import torch

from caldera.utils import long_isin
from caldera.utils import reindex_tensor
from caldera.utils import same_storage
from caldera.utils.nx import nx_is_directed
from caldera.utils.nx.types import DirectedGraph


def np_or_tensor_size(arr: Union[torch.tensor, np.ndarray]) -> int:
    if issubclass(arr.__class__, torch.Tensor):
        return arr.nelement()
    elif issubclass(arr.__class__, np.ndarray):
        return arr.size
    else:
        raise ValueError("Must be a {} or {}".format(np.ndarray, torch.Tensor))


# TODO: apply freeze method
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
        node_attr: torch.FloatTensor,
        edge_attr: torch.FloatTensor,
        global_attr: torch.FloatTensor,
        edges: torch.LongTensor,
        requires_grad: Optional[bool] = None,
    ):
        """Blank.

        :param node_attr:
        :param edge_attr:
        :param global_attr:
        :param edges:
        :param requires_grad:
        """
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

        if self.edges.shape[0]:
            n_edges = self.edges.shape[1]
        else:
            n_edges = 0
        if not n_edges == self.e.shape[0]:
            raise RuntimeError(
                "Number of edges {} must match number of edge attributes {}".format(
                    n_edges, self.e.shape[0]
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
            keys = [k for k, _ in self._tensors]
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

    def detach(self):
        return self.apply(lambda x: x.detach())

    def to(self, device: str, *args, **kwargs):
        return self.apply(lambda x: x.to(device, *args, **kwargs))

    @property
    def _tensors(self):
        for k in self.__slots__:
            if not k.startswith("__"):
                v = getattr(self, k)
                if torch.is_tensor(v):
                    yield k, v

    def share_storage(
        self, other: GraphData, return_dict: Optional[bool] = False
    ) -> Union[Dict[str, bool], bool]:
        """Check if this data shares storage with another data.

        :param other: The other GraphData object.
        :param return_dict: if true, return dictionary of which tensors share the same
            storage. Else returns true if any tensors share the same storage.
        :return:
        """
        d = {}
        for k, _ in self._tensors:
            a = getattr(self, k)
            b = getattr(other, k)
            if 0 in a.shape or 0 in b.shape:
                shares_storage = False
            else:
                shares_storage = same_storage(a, b)
            if return_dict:
                d[k] = shares_storage
            elif shares_storage:
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
    def num_edges(self):
        return self.edges.shape[1]

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

    # def _mask_fields(self, masks: Dict[str, torch.tensor]):
    #     for m in masks:
    #         if m not in self.__slots__:
    #             raise RuntimeError("{} is not a valid field".format(m))
    #     masked_fields = []
    #     for field in self.__slots__:
    #         if field not in masks or masks[field] is None:
    #             masked_fields.append(getattr(self, field))
    #         else:
    #             masked_fields.append(getattr(self, field)[:, masks[field]])
    #     return masked_fields
    #
    # def mask(self, node_mask, edge_mask, global_mask, invert: bool = False):
    #     d = {"x": node_mask, "e": edge_mask, "g": global_mask}
    #     if invert:
    #         d = {k: ~v for k, v in d.items()}
    #     return self.__class__(*self._mask_fields(d))

    @staticmethod
    def _apply_mask(
        arr: torch.Tensor,
        mask: torch.BoolTensor,
        detach: bool,
        as_view: bool,
        dim: int = 0,
    ):
        ret = arr
        if detach:
            ret = arr.detach()

        if mask is None:
            if not as_view:
                ret = arr.clone()
        else:
            if as_view:
                mask = torch.where(mask)
            if isinstance(mask, tuple) or dim == 0:
                ret = arr[mask]
            elif dim == 1:
                ret = arr[:, mask]
            else:
                raise ValueError("dim must be 0 or 1")
        return ret

    def _validate_masks(
        self,
        node_mask: Optional[torch.BoolTensor],
        edge_mask: Optional[torch.BoolTensor],
    ):
        if node_mask is not None and not node_mask.ndim == 1:
            raise ValueError("Node mask must be 1 dimensional")
        if edge_mask is not None and not edge_mask.ndim == 1:
            raise ValueError("Edge mask must be 1 dimensional")
        if node_mask is not None and not node_mask.dtype == torch.bool:
            raise ValueError(
                "Node mask must be tensor.BoolTensor, not " + str(node_mask.dtype)
            )
        if edge_mask is not None and not edge_mask.dtype == torch.bool:
            raise ValueError(
                "Edge mask must be tensor.BoolTensor, not " + str(edge_mask.dtype)
            )

    def _mask_dispatch_reindex_edges(self, edges, node_mask):
        # remap node indices in edges
        if node_mask is not None and not torch.all(node_mask):
            nidx = torch.where(node_mask)[0]
            _, edges = reindex_tensor(nidx, edges)
        return edges

    def _mask_dispatch_constructor(self, new_inst: bool, *args, **kwargs):
        if new_inst:
            constructor = self.__class__
        else:

            def constructor(*args, **kwargs):
                self.__init__(*args, **kwargs)
                return self

        masked_data = constructor(*args, **kwargs)
        masked_data.debug()
        return masked_data

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
        return self._mask_dispatch_constructor(new_inst, x, e, g, edges)

    def apply_edge_mask_(self, mask: torch.BoolTensor) -> GraphData:
        """In place version of :meth:`caldera.data.GraphData.apply_edge_mask`

        :param mask: boolean mask
        :return: self
        """
        return self._mask_dispatch(
            None, mask, as_view=True, detach=False, new_inst=False
        )

    def apply_edge_mask(self, mask: torch.BoolTensor) -> GraphData:
        """Apply edge mask to the graph, returning a new :class:`GraphData`
        instance.

        :param mask: boolean mask
        :return: a new :class:`GraphData` instance.
        """
        return self._mask_dispatch(
            None, mask, as_view=False, detach=True, new_inst=True
        )

    def _get_edge_mask_from_nodes(self, nodes: torch.LongTensor):
        idx = self._gather(self._node_to_edge_idx, nodes)
        mask = torch.BoolTensor([True] * self.num_edges)
        mask[idx] = False
        return mask

    def _apply_node_mask_dispatch(
        self, node_mask, as_view: bool, detach: bool, new_inst: bool
    ):
        nidx = torch.where(~node_mask)[0]
        edge_mask = ~torch.any(
            long_isin(self.edges.flatten(), nidx, invert=False).view(2, -1), 0
        )
        return self._mask_dispatch(
            node_mask, edge_mask, as_view=as_view, detach=detach, new_inst=new_inst
        )

    def apply_node_mask_(self, node_mask: torch.BoolTensor) -> GraphData:
        """In place version of :meth:`caldera.data.GraphData.apply_node_mask`.

        :param node_mask: boolean mask
        :return: self
        """
        return self._apply_node_mask_dispatch(
            node_mask, as_view=True, detach=False, new_inst=False
        )

    def apply_node_mask(self, node_mask: torch.BoolTensor) -> GraphData:
        """Apply node mask to the graph, returning a new :class:`GraphData`
        instance, removing any edges if necessary. Note this will reindex any
        indices in the data (e.g. `edges`)

        :param mask: boolean mask
        :return: a new :class:`GraphData` instance.
        """
        return self._apply_node_mask_dispatch(
            node_mask, as_view=False, detach=True, new_inst=True
        )

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
        return self.apply(
            lambda x: torch.empty_like(x, *emtpy_like_args, **emtpy_like_kwargs).copy_(
                x, non_blocking=non_blocking
            )
        )

    # TODO: nbexamples
    # TODO: handle undirected and hypergraphs
    # TODO: check that features are NUMPY rather than TORCH

    @staticmethod
    def _get_nx_feature_size(g: DirectedGraph, key: str, global_key: str = None):
        def collect_feature_shapes(datalist):
            shapes = []
            for x in datalist:
                data = x[-1]
                if key not in data:
                    shape = None
                else:
                    shape = tuple(data[key].shape)
                shapes.append(shape)
            return shapes

        def get_feature_shape(arr, m):
            shapes = set(collect_feature_shapes(arr))
            if arr is ...:
                return 0
            if len(shapes) > 1:
                if None in shapes:
                    raise RuntimeError(
                        "{x} features shapes must be the same. Found {shapes}. "
                        "At least one {x} is missing a feature '{key}'.".format(
                            x=m, shapes=shapes, key=key
                        )
                    )
                else:
                    raise RuntimeError(
                        "{x} features for '{key}' must be the same. Found {shapes}.".format(
                            x=m, shapes=shapes, key=key
                        )
                    )
            elif len(shapes) == 0:
                return 0
            elif list(shapes)[0] is None:
                return 0
            return reduce(lambda a, b: a * b, list(shapes)[0])

        n_feat = get_feature_shape(g.nodes(data=True), "node")
        e_feat = get_feature_shape(g.edges(data=True), "edge")
        g_feat = get_feature_shape(
            g.globals(data=True, global_key=global_key), "global"
        )

        return n_feat, e_feat, g_feat

    # TODO: validate features are all same size
    # TODO: estimate sizes of features

    @classmethod
    def from_networkx(
        cls,
        g: DirectedGraph,
        n_node_feat: Optional[int] = None,
        n_edge_feat: Optional[int] = None,
        n_glob_feat: Optional[int] = None,
        *,
        feature_key: str = "features",
        global_attr_key: str = None,
        requires_grad: Optional[bool] = None,
        dtype: str = torch.float32,
    ):
        """Create a new :class:`GraphData` from a networkx graph.

        :param g:
        :param n_node_feat:
        :param n_edge_feat:
        :param n_glob_feat:
        :param feature_key: The key to look for data in node, edge, and global data.
        :param global_attr_key: Key to look for global data.
        :return:
        """
        if not isinstance(g, nx.Graph) or not nx_is_directed(g):
            raise TypeError(
                "Graph must be a directed graph instance, not a '{}'. Convert to directed graph first.".format(
                    g.__class__.__name__
                )
            )
        gdata = g.get_global(global_attr_key)
        _n_node_feat, _n_edge_feat, _n_glob_feat = cls._get_nx_feature_size(
            g, feature_key, global_attr_key
        )
        n_node_feat = n_node_feat or _n_node_feat
        n_edge_feat = n_edge_feat or _n_edge_feat
        n_glob_feat = n_glob_feat or _n_glob_feat
        n_node_size, n_edge_size, n_glob_size = (
            g.number_of_nodes(),
            g.number_of_edges(),
            1,
        )
        node_attr = np.empty((n_node_size, n_node_feat))
        edge_attr = np.empty((n_edge_size, n_edge_feat))
        glob_attr = np.empty((n_glob_size, n_glob_feat))

        nodes = sorted(list(g.nodes(data=True)))
        ndict = {}
        # TODO: support n dim tensors (but they get flattened anyways...)
        for i, (n, ndata) in enumerate(nodes):
            node_attr[i] = ndata.get(feature_key, np.array([])).flatten()
            ndict[n] = i

        # TODO: method to change dtype?
        edges = np.empty((2, n_edge_size), dtype=np.float)
        for i, (n1, n2, edata) in enumerate(g.edges(data=True)):
            edges[:, i] = np.array([ndict[n1], ndict[n2]])
            edge_attr[i] = edata.get(feature_key, np.array([])).flatten()

        if feature_key in gdata:
            glob_attr[0] = gdata.get(feature_key, np.array([])).flatten()

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
        global_attr_key: str = None,
        graph_type: Type[DirectedGraph] = nx.OrderedMultiDiGraph,
    ) -> DirectedGraph:
        g = graph_type()
        for n, ndata in enumerate(self.x):
            g.add_node(n, **{feature_key: ndata})
        g.ordered_edges = []
        for i, e in enumerate(self.edges.T):
            n = g.add_edge(e[0].item(), e[1].item(), **{feature_key: self.e[i]})
            g.ordered_edges.append((e[0].item(), e[1].item(), n))
        g.set_global({feature_key: self.g.clone()}, global_attr_key)
        return g

    def __repr__(self):
        return "<{cls} size(n,e,g)={size} features(n,e,g)={shape}>".format(
            cls=self.__class__.__name__,
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
        cls,
        n_feat: int,
        e_feat: int,
        g_feat: int,
        requires_grad: Optional[bool] = None,
        min_nodes: int = None,
        max_nodes: int = None,
        min_edges: int = None,
        max_edges: int = None,
    ) -> GraphData:
        # initialize defaults
        if min_nodes is None and max_nodes is None:
            min_nodes = 1
            max_nodes = 20
        elif min_nodes is None and max_nodes is not None:
            min_nodes = min(1, max_nodes)
        elif min_nodes is not None and max_nodes is None:
            max_nodes = max(min_nodes, 10)

        if min_edges is None and max_edges is None:
            min_edges = 1
            max_edges = max(min_edges, int(0.5 * max_nodes))
        elif min_edges is None and max_edges is not None:
            min_edges = min(1, max_edges)
        elif min_edges is not None and max_edges is None:
            max_edges = max(min_edges, int(0.5 * max_nodes))

        n_nodes = torch.randint(min_nodes, max_nodes + 1, torch.Size([])).item()
        n_edges = torch.randint(min_edges, max_edges + 1, torch.Size([])).item()

        edges = torch.empty((2, 0), dtype=torch.long)
        if n_nodes:
            edges = torch.randint(0, n_nodes, torch.Size([2, n_edges]))
        else:
            n_edges = 0
        return cls(
            torch.randn(n_nodes, n_feat),
            torch.randn(n_edges, e_feat),
            torch.randn(1, g_feat),
            edges,
            requires_grad=requires_grad,
        )

    # TODO: view
    def view(
        self,
        x_slice: Optional[slice] = None,
        e_slice: Optional[slice] = None,
        g_slice: Optional[slice] = None,
    ) -> GraphData:
        if x_slice is None:
            x_slice = slice(None, None, None)
        if e_slice is None:
            e_slice = slice(None, None, None)
        if g_slice is None:
            g_slice = slice(None, None, None)
        return self.__class__(
            self.x[:, x_slice], self.e[:, e_slice], self.g[:, g_slice], self.edges
        )

    def index_nodes(self, idx: torch.LongTensor) -> GraphData:
        """Apply index to nodes."""
        cloned = self.detach().clone()
        cloned.index_nodes_(idx)
        return cloned

    def index_nodes_(self, idx: torch.LongTensor) -> None:
        """In place version of :meth:`caldera.data.GraphData.index_nodes`"""
        assert idx.ndim == 1
        assert idx.shape[0] == self.num_nodes
        _, edges = reindex_tensor(idx, self.edges)
        x = self.x[idx]
        self.edges = edges
        self.x = x

    def index_edges(self, idx: torch.LongTensor) -> GraphData:
        """Apply index to nodes."""
        cloned = self.detach().clone()
        cloned.index_edges_(idx)
        return cloned

    def index_edges_(self, idx: torch.LongTensor) -> None:
        """In place version of :meth:`caldera.data.GraphData.index_edges`"""
        assert idx.ndim == 1
        assert idx.shape[0] == self.num_edges
        self.edges = self.edges[:, idx]
        self.e = self.e[idx]

    def shuffle_nodes(self) -> GraphData:
        cloned = self.detach().clone()
        cloned.shuffle_nodes_()
        return cloned

    def shuffle_edges(self) -> GraphData:
        cloned = self.detach().clone()
        cloned.shuffle_edges_()
        return cloned

    def shuffle_nodes_(self) -> None:
        idx = torch.randperm(self.num_nodes)
        self.index_nodes_(idx)

    def shuffle_edges_(self) -> None:
        idx = torch.randperm(self.num_edges)
        self.index_edges_(idx)

    def shuffle_(self) -> None:
        self.shuffle_edges_()
        self.shuffle_nodes_()

    def shuffle(self) -> GraphData:
        cloned = self.detach().clone()
        cloned.shuffle_()
        return cloned

    def reverse(self) -> GraphData:
        cloned = self.detach().clone()
        cloned.reverse_()
        return cloned

    def reverse_(self):
        self.edges = self.edges.flip(1)

    def nelement(self) -> int:
        """Return total number of elements in the.

        :class:`caldera.data.GraphData` instance
        """
        x = 0
        for _, t in self._tensors:
            if hasattr(t, "nelement"):
                x += t.nelement()
        return x

    def memsize(self):
        """Return total number of bytes in the.

        :class:`caldera.data.GraphData` instance
        """
        x = 0
        for _, t in self._tensors:
            if hasattr(t, "nelement"):
                x += t.element_size() * t.nelement()
        return x

    def density(self):
        """Return density of the graph."""
        return self.num_edges / (self.num_nodes * (self.num_nodes - 1))

    def _get_edge_dict(self):
        src, dest = self.edges.tolist()
        edge_dict = {}
        for _src, _dest in zip(src, dest):
            edge_dict.setdefault(_src, list())
            edge_dict[_src].append(_dest)
        return edge_dict

    def info(self):
        msg = "{}(\n".format(self.__class__.__name__)
        msg += "  n_nodes: {}\n".format(self.num_nodes)
        msg += "  n_edges: {}\n".format(self.num_edges)
        msg += "  feat_shape: {}\n".format(tuple(self.shape))
        msg += "  size: {}\n".format(tuple(self.size))
        msg += "  nelements: {}\n".format(self.nelement())
        msg += "  bytes: {}\n".format(self.memsize())
        msg += ")"
        return msg

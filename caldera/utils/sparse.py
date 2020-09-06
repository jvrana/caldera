from typing import Optional
from typing import Type

import torch
from scipy.sparse import coo_matrix

from .indexing import SizeType
from .indexing import unroll_index


def torch_coo_to_scipy_coo(m: torch.sparse.FloatTensor) -> coo_matrix:
    """Convert torch :class:`torch.sparse.FloatTensor` tensor to.

    :class:`scipy.sparse.coo_matrix`
    """
    data = m.values().numpy()
    indices = m.indices()
    return coo_matrix((data, (indices[0], indices[1])), tuple(m.size()))


def scatter_indices(indices: torch.LongTensor, shape: SizeType):
    """Unroll the coo indices using the provided shape.

    .. code-block::

        indices = torch.tensor([
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 4]
        ])
        shape = (3, 2)
        print(scatter_indices(indices, shape))

        # tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
        #  0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        # [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4,
        #  2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
        # [4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4,
        #  4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4],
        # [0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1,
        #  0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1]])

    :param indices:
    :param shape:
    :return:
    """
    if not shape:
        return indices
    idx = torch.stack(unroll_index(shape))

    a_repeat = [1] * indices.ndim
    a_repeat[-1] = idx.shape[-1]
    b_repeat = [1] * indices.ndim
    b_repeat[-1] = indices.shape[-1]

    a = torch.repeat_interleave(indices, idx.shape[-1], dim=1)
    b = idx.repeat(b_repeat)
    return torch.cat((a, b))


def _expand_idx(idx):
    if idx.ndim == 1:
        idx = idx.unsqueeze(0)
        idx = torch.cat((torch.zeros_like(idx), idx))
    return idx


def _coo_tensor(
    indices: torch.LongTensor,
    source: torch.Tensor,
    size: Optional[SizeType] = None,
    dtype: Optional[Type] = None,
    **kwargs
):
    if size is not None:
        kwargs["size"] = size
    if dtype is None:
        kwargs["dtype"] = source.dtype
    else:
        kwargs["dtype"] = dtype
    if size is not None:
        kwargs = dict(dtype=dtype, size=size)
    else:
        kwargs = dict(dtype=dtype)
    return torch.sparse_coo_tensor(indices, source, **kwargs)


# TODO: infer size from index sizes
def scatter_coo(
    indices: torch.LongTensor,
    source: torch.FloatTensor,
    size: Optional[SizeType] = None,
    expand: bool = False,
    dtype: Optional[Type] = None,
) -> torch.sparse.FloatTensor:
    """Scatter the provided source tensor to the provided indices.

    :param indices:
    :param source:
    :return:
    """

    indices = _expand_idx(indices)

    if not torch.is_tensor(source):
        source = torch.tensor(source)

    if expand:
        shape = source.shape
        # r = prod(shape[:-1]) * indices.shape[1]
        r = indices.shape[1]
        flattened = source.view(-1).repeat(r)
    else:
        shape = source.shape[1:]
        flattened = source.view(-1)

    if size is not None and size[-1] is ...:
        if not len(size) - 1 == indices.shape[0]:
            raise ValueError(
                "Provided dims ({}) must match number of index dims ({})".format(
                    len(size) - 1, indices.shape[0]
                )
            )
        size = tuple(list(size)[:-1]) + shape

    sidx = scatter_indices(indices, shape)
    return _coo_tensor(sidx, flattened, size=size, dtype=dtype)


#
# def scatter_coo_fill(
#     indices: torch.LongTensor,
#     source: torch.FloatTensor,
#     size: Optional[SizeType] = None,
#     dtype: Optional[Type] = None,
# ) -> torch.sparse.FloatTensor:
#     """Fill sparse coo matrix with the provided tensor at the provided indices.
#
#     :param indices:
#     :param source:
#     :return:
#     """
#     indices = _expand_idx(indices)
#     source = torch.tensor(source)
#     sidx = scatter_indices(indices, source.shape)
#     if size is not None and size[-1] is ...:
#         size = tuple(list(size)[:-1])
#         if torch.is_tensor():
#             size += source.shape
#     return _coo_tensor(
#         sidx, source.view(-1).repeat(indices.shape[1]), size=size, dtype=dtype
#     )

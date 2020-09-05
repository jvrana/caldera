r"""
Utils (:mod:`caldera.utils`)
==============================

.. currentmodule:: caldera.utils

Caldera utility functions.

.. autosummary::
    :toctree: generated/
"""


from ._dict_join import dict_join
from ._iteration import _first
from ._iteration import pairwise
from caldera.utils.indexing import reindex_tensor
from caldera.utils.indexing import unravel_index
from caldera.utils.sparse import scatter_coo
from caldera.utils.sparse import scatter_indices
from caldera.utils.sparse import torch_coo_to_scipy_coo
from caldera.utils.tensor import deterministic_seed
from caldera.utils.tensor import long_isin
from caldera.utils.tensor import same_storage
from caldera.utils.tensor import stable_arg_sort_long
from caldera.utils.tensor import tensor_is_empty
from caldera.utils.tensor import torch_scatter_group


__all__ = [
    'reindex_tensor',
    'unravel_index',
    'scatter_coo',
    'scatter_indices',
    'torch_coo_to_scipy_coo',
    'deterministic_seed',
    'long_isin',
    'same_storage',
    'stable_arg_sort_long',
    'tensor_is_empty',
    'torch_scatter_group',
    'dict_join',
    'pairwise',
    '_first',
]

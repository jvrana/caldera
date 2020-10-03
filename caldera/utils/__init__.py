r"""

Caldera utility functions.

.. autosummary::
    :toctree: _generated/

    dict_join
    # pairwise


Indexing
--------

.. autosummary::
    :toctree: _generated/

    reindex_tensor
    unravel_index

Tensor
------

Utilities for :class:`torch.Tensor`

.. autosummary::
    :toctree: _generated/

    scatter_coo
    scatter_indices
    torch_coo_to_scipy_coo
    deterministic_seed
    long_isin
    same_storage
    stable_arg_sort_long
    tensor_is_empty
    torch_scatter_group

Functional
----------

Functional programming module.

.. autosummary::
    :toctree: _generated/

    functional

Networkx Utilities
------------------

Extra :mod:`networkx` utilities

.. autosummary::
    :toctree: _generated/

    nx
"""
from ._dict_join import dict_join
from ._iteration import _first
from ._iteration import pairwise
from caldera.utils.indexing import reindex_tensor
from caldera.utils.indexing import unravel_index
from caldera.utils.np import replace_nan_with_inf
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
    "reindex_tensor",
    "unravel_index",
    "scatter_coo",
    "scatter_indices",
    "torch_coo_to_scipy_coo",
    "deterministic_seed",
    "long_isin",
    "same_storage",
    "stable_arg_sort_long",
    "tensor_is_empty",
    "torch_scatter_group",
    "dict_join",
    "pairwise",
    "_first",
    "replace_nan_with_inf",
]

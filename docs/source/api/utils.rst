Utils
=====

.. currentmodule:: caldera.utils

Caldera utility functions.

.. autosummary::
    :toctree: ../generated/

    dict_join
    # pairwise


Indexing
--------

.. autosummary::
    :toctree: ../generated/

    reindex_tensor
    unravel_index

Tensor
------

Utilities for :class:`torch.Tensor`

.. autosummary::
    :toctree: ../generated/

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
    :toctree: ../generated/
    :recursive:

    functional

Networkx Utilities
------------------

Extra :mod:`networkx` utilities

.. autosummary::
    :toctree: ../generated/
    :recursive:

    nx
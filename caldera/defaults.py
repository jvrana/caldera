r"""
Defaults (:mod:`caldera.defaults`)
==================================

Caldera defaults.
"""

import torch


class CalderaDefaults:

    activation = torch.nn.LeakyReLU
    nx_global_key = "_global_attr"

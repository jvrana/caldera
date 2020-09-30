"""
mp
==

Multiprocessing utils
"""
from ._decorators import multiprocess
from ._mp_tools import run_with_pool

__all__ = ["run_with_pool", "multiprocess"]

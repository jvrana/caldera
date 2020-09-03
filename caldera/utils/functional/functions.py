"""functions.py.

Common functions
"""
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple
from typing import TypeVar

from caldera.utils import functional as Fn

K = TypeVar("K")
V = TypeVar("V")


def group_inner_dict(datalist: List[Dict[K, V]]) -> Dict[K, List[V]]:
    return Fn.compose(
        Fn.map_each(lambda x: x.items()),
        Fn.chain_each(),
        Fn.group_each_by_key(lambda x: x[0]),
        Fn.map_each(lambda x: (x[0], [_x[1] for _x in x[1]])),
        dict,
    )(
        datalist
    )  # from a list of dictionaries, List[Dict] -> Dict[str, List]

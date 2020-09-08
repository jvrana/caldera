from typing import Callable
from typing import TypeVar

import networkx as nx

from ._base import NetworkxTransformBase

T = TypeVar("T")


class NetworkxApply(NetworkxTransformBase):
    def __init__(self, func: Callable[[nx.Graph], T]):
        self.func = func

    def transform(self, g: nx.Graph) -> T:
        return self.func(g)

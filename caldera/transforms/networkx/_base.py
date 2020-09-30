"""_base.py.

Transform base class
"""
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Generator
from typing import List
from typing import overload
from typing import Tuple
from typing import Union

import networkx as nx

from ._types import _G


class NetworkxTransformBase(ABC):
    def __repr__(self):
        return "{}".format(self.__class__.__name__)

    @abstractmethod
    def transform(self, data: _G) -> _G:
        pass

    def generate(
        self, datalist: Union[List[_G], Tuple[_G, ...], Generator[_G, None, None]]
    ) -> Generator[_G, None, None]:
        for data in datalist:
            yield self.transform(data)

    @overload
    def __call__(self, data: List[_G]) -> List[_G]:
        ...

    @overload
    def __call__(self, data: Tuple[_G]) -> Tuple[_G]:
        ...

    @overload
    def __call__(self, data: Generator[_G, None, None]) -> Generator[_G, None, None]:
        ...

    @overload
    def __call__(self, data: _G) -> _G:
        ...

    def __call__(self, data: _G) -> _G:
        if isinstance(data, list):
            transformed = list(self.generate(data))
        elif isinstance(data, tuple):
            transformed = tuple(self.generate(data))
        elif isinstance(data, nx.Graph):
            transformed = list(self.generate([data]))[0]
        else:
            transformed = self.generate(data)
        return transformed


TransformCallable = Union[Callable, NetworkxTransformBase]

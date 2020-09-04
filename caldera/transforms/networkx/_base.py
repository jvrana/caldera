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
from typing import TypeVar
from typing import Union

import networkx as nx


T = TypeVar("T")


class NetworkxTransformBase(ABC):
    def __repr__(self):
        return "{}".format(self.__class__.__name__)

    @abstractmethod
    def transform(self, data):
        pass

    def generate(self, data):
        for d in data:
            yield self.transform(d)

    @overload
    def __call__(self, data: List[T]) -> List[T]:
        ...

    @overload
    def __call__(self, data: Tuple[T]) -> Tuple[T]:
        ...

    @overload
    def __call__(self, data: Generator[T, None, None]) -> Generator[T, None, None]:
        ...

    @overload
    def __call__(self, data: T) -> T:
        ...

    def __call__(self, data: T) -> T:
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

"""_base.py.

Transform base class
"""
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import overload
from typing import Union

from caldera.data import GraphBatch
from caldera.data import GraphData


class TransformBase(ABC):
    @abstractmethod
    def transform(self, data):
        pass

    @overload
    def __call__(self, data: GraphBatch) -> GraphBatch:
        ...

    def __call__(self, data: GraphData) -> GraphData:
        return self.transform(data)

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


TransformCallable = Union[Callable, TransformBase]

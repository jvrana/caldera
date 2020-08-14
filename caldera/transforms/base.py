"""
base.py

Transform base class
"""
from abc import ABC
from typing import Callable, Union


class TransformBase(ABC):

    def __init__(self):
        pass

    def __call__(self, data):
        return self.transform(data)

    def __repr__(self):
        return "{}".format(
            self.__class__.__name__
        )


TransformCallable = Union[Callable, TransformBase]
from .base import TransformBase, TransformCallable
from caldera.data import GraphData, GraphBatch
from typing import Union


class ToFullyConnected(TransformBase):

    def __call__(self, data: Union[GraphData, GraphBatch]):
        pass
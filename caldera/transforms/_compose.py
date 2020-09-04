"""_compose.py.

Transform that composes other transforms. Exactly the same as
TorchVision interface.
"""
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Union

from ._base import TransformCallable

T = TypeVar("T")


class Compose:
    def __init__(
        self, transforms: Union[List[TransformCallable], Tuple[TransformCallable, ...]]
    ):
        """Composes several transforms together.

        :param transforms:
        """
        self.transforms = transforms

    def __call__(self, data: T) -> T:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        sep = "\n   "
        format_string = self.__class__.__name__ + "("
        format_string += sep.join([str(t) for t in self.transforms])
        format_string += "\n)"
        return format_string

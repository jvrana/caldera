from itertools import tee
from typing import Any
from typing import Callable
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import TypeVar

from torch.utils.data import DataLoader

from pyrographnets.data import GraphBatch
from pyrographnets.data import GraphData
from pyrographnets.utils import _first


T = TypeVar("T")


def collate(data_list):
    if isinstance(data_list[0], tuple):
        if issubclass(type(data_list[0][0]), GraphData):
            return tuple(
                [collate([x[i] for x in data_list]) for i in range(len(data_list[0]))]
            )
        else:
            raise RuntimeError(
                "Cannot collate {}({})({})".format(
                    type(data_list), type(data_list[0]), type(data_list[0][0])
                )
            )
    return GraphBatch.from_data_list(data_list)


class GraphDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super().__init__(dataset, batch_size, shuffle, collate_fn=collate, **kwargs)

    def first(self, *args, **kwargs):
        return _first(tee(self(*args, **kwargs))[0])

    def __call__(
        self,
        device: Optional[str] = None,
        f: Optional[Callable[[GraphBatch], T]] = None,
        send_to_device_before_apply: bool = True,
    ) -> Generator[T, None, None]:
        """Create a new generator. Optionall apply some function to each item
        or send the item to a device. In cases where both are defined, items
        are first sent to device and then the function is applied, unless
        argument "send_to_device_before_apply" is set to False.

        :param device: optional device to send each item to
        :param f: function to apply to each item
        :param send_to_device_before_apply: if True (default), send to device before applying function (if applicable)
        :return: generator of items
        """
        for d in self:
            if device is not None and f is not None:
                if send_to_device_before_apply:
                    yield f(d.to(device))
                else:
                    yield f(d).to(device)
            elif device is not None:
                yield d.to(device)
            elif f is not None:
                yield f(d)
            else:
                yield d

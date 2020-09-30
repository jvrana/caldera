from itertools import tee
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils import _first

T = TypeVar("T")


@overload
def collate_list(data_list: List[GraphBatch]) -> GraphBatch:
    ...


def collate_list(data_list: List[GraphData]) -> GraphBatch:
    return GraphBatch.from_data_list(data_list)


@overload
def collate_zip(data_list: List[Tuple[GraphBatch, ...]]) -> Tuple[GraphBatch, ...]:
    ...


def collate_zip(data_list: List[Tuple[GraphData, ...]]) -> Tuple[GraphBatch, ...]:
    if issubclass(type(data_list[0][0]), GraphData):
        return tuple(
            [collate_list([x[i] for x in data_list]) for i in range(len(data_list[0]))]
        )
    else:
        raise RuntimeError(
            "Cannot collate {}({})({})".format(
                type(data_list), type(data_list[0]), type(data_list[0][0])
            )
        )


class GraphDataLoader(DataLoader):
    def __init__(
        self,
        *dataset: Union[List[GraphBatch], Dataset],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs
    ):
        if len(dataset) > 1:
            dataset = list(zip(*dataset))
            self.zipped = True
            collate = collate_zip
        else:
            dataset = dataset[0]
            collate = collate_list
        super().__init__(dataset, batch_size, shuffle, collate_fn=collate, **kwargs)

    def first(self, *args, **kwargs):
        return _first(tee(self(*args, **kwargs))[0])

    def mem_sizes(self):
        return torch.IntTensor([d.memsize() for d in self.dataset])

    def __call__(
        self,
        device: Optional[str] = None,
        f: Optional[Callable[[GraphBatch], T]] = None,
        send_to_device_before_apply: bool = True,
        limit_mem_size: Optional[int] = None,
    ) -> Generator[T, None, None]:
        """Create a new generator. Optionally apply some function to each item
        or send the item to a device. In cases where both are defined, items
        are first sent to device and then the function is applied, unless
        argument "send_to_device_before_apply" is set to False.

        :param device: optional device to send each item to
        :param f: function to apply to each item
        :param send_to_device_before_apply: if True (default), send to device before applying function (if applicable)
        :param limit_mem_size: Limit memory size of the batches. Warning, this will introduce data bias such that
            large graphs are less likely to appear together. Consider using this carefully.
        :return: generator of items
        """
        for d in self:
            if device is not None and f is not None:
                if send_to_device_before_apply:
                    ret = f(d.to(device))
                else:
                    ret = f(d).to(device)
            elif device is not None:
                ret = d.to(device)
            elif f is not None:
                ret = f(d)
            else:
                ret = d
            if limit_mem_size is not None:
                if ret.memsize() > limit_mem_size:
                    continue
            yield ret

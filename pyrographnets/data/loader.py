from torch.utils.data import DataLoader

from pyrographnets.data import GraphData, GraphBatch
from itertools import tee
from pyrographnets.utils import _first


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

    def first(self):
        return _first(tee(self)[0])

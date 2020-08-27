from typing import List
from typing import Optional
from typing import overload

import torch
from torch.utils.data import Dataset

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.transforms import TransformCallable


class GraphDataset(Dataset):
    """Graph dataset."""

    def __init__(
        self, datalist: List[GraphData], transform: Optional[TransformCallable] = None
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert isinstance(datalist, list)
        self.datalist = datalist
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def mem_sizes(self):
        return torch.IntTensor([d.memsize() for d in self.dataset])

    def get(self, idx, transform: Optional[TransformCallable] = None):
        transform = transform or self.transform
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
            idx = [idx]
            one_item = True
        else:
            one_item = False

        if isinstance(idx, slice):
            samples = self.datalist[idx]
        else:
            samples = []
            for i in idx:
                sample = self.datalist[i]
                if transform:
                    sample = transform(sample)
                samples.append(sample)

        if one_item:
            samples = samples[0]
        return samples

    @overload
    def __getitem__(self, idx: torch.LongTensor) -> List[GraphData]:
        ...

    @overload
    def __getitem__(self, idx: List[int]) -> List[GraphData]:
        ...

    def __getitem__(self, idx: int) -> GraphData:
        return self.get(idx)


class GraphBatchDataset(GraphDataset):
    """Dataset that loads a :class:`caldera.data.GraphData` list into.

    :class:`caldera.data.GraphBatch` instances, and then performs transform (if
    provided).
    """

    @overload
    def __getitem__(self, idx: torch.LongTensor) -> GraphBatch:
        ...

    @overload
    def __getitem__(self, idx: List[int]) -> GraphData:
        ...

    def __getitem__(self, idx: int) -> GraphBatch:
        samples = self.get(idx, transform=None)
        if isinstance(samples, GraphData):
            samples = [samples]
        batch = GraphBatch.from_data_list(samples)
        if self.transform:
            batch = self.transform(batch)
        return batch

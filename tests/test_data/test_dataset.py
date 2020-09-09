import pytest
import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.dataset import GraphBatchDataset
from caldera.dataset import GraphDataset
from caldera.transforms import RandomEdgeMask
from caldera.transforms import RandomNodeMask


@pytest.mark.incremental
@pytest.mark.parametrize(
    ("Dataset", "DataType"),
    [(GraphDataset, GraphData), (GraphBatchDataset, GraphBatch)],
    ids=["GraphDataset", "GraphBatchDataset"],
)
class TestDataset:
    @pytest.mark.parametrize(
        "random_data_list",
        [(100, GraphData), (1, GraphData), (0, GraphData)],
        indirect=True,
        ids=lambda x: str(x[0]) + "_" + x[1].__name__,
    )
    def test_init_dataset(self, Dataset, DataType, random_data_list):
        data_set = Dataset(random_data_list)
        assert len(data_set) == len(random_data_list)
        for data in data_set:
            assert isinstance(data, DataType)

    @pytest.mark.parametrize(
        "random_data_list",
        [(300, GraphData)],
        indirect=True,
        ids=lambda x: str(x[0]) + "_" + x[1].__name__,
    )
    @pytest.mark.parametrize(
        "idx",
        [
            0,
            1,
            10,
            [0],
            [0, 1, 20],
            torch.LongTensor([0, 1, 20]),
            torch.IntTensor([0, 1, 2, 3, 4]),
        ],
    )
    def test_indexing_data_set(self, Dataset, DataType, idx, random_data_list):
        data_set = Dataset(random_data_list)
        assert len(data_set) == len(random_data_list)
        selected = data_set[idx]

    @pytest.mark.parametrize(
        "random_data_list",
        [(300, GraphData)],
        indirect=True,
        ids=lambda x: str(x[0]) + "_" + x[1].__name__,
    )
    @pytest.mark.parametrize(
        "idx", [slice(None, None, None), slice(None, 10, None), slice(None, None, 2)]
    )
    def test_slicing_dataset(self, Dataset, DataType, idx, random_data_list):
        data_set = Dataset(random_data_list)
        assert len(data_set) == len(random_data_list)
        selected = data_set[idx]
        assert selected


@pytest.mark.incremental
@pytest.mark.parametrize(
    ("Dataset", "DataType"),
    [(GraphDataset, GraphData), (GraphBatchDataset, GraphBatch)],
    ids=["GraphDataset", "GraphBatchDataset"],
)
class TestDatasetTransform:
    @pytest.mark.parametrize(
        "random_data_list",
        [(300, GraphData)],
        indirect=True,
        ids=lambda x: str(x[0]) + "_" + x[1].__name__,
    )
    def test_dataset_with_node_mask(self, Dataset, DataType, random_data_list):
        transform = RandomNodeMask(0.2)
        dataset1 = Dataset(random_data_list, transform=transform)
        dataset2 = Dataset(random_data_list, transform=None)
        n1, n2 = 0, 0
        for data in dataset1:
            n1 += data.num_nodes
        for data in dataset2:
            n2 += data.num_nodes
        assert n1 < n2

    @pytest.mark.parametrize(
        "random_data_list",
        [(300, GraphData)],
        indirect=True,
        ids=lambda x: str(x[0]) + "_" + x[1].__name__,
    )
    def test_dataset_with_edge_mask(self, Dataset, DataType, random_data_list):
        transform = RandomEdgeMask(0.2)
        dataset1 = Dataset(random_data_list, transform=transform)
        dataset2 = Dataset(random_data_list, transform=None)
        n1, n2 = 0, 0
        for data in dataset1:
            n1 += data.num_edges
        for data in dataset2:
            n2 += data.num_edges
        assert n1 < n2

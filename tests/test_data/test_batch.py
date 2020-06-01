from pyrographnets.data import GraphBatch
import torch
import pytest
from pyrographnets.data.utils import random_data


def test_random_data():
    random_data(5, 4, 3)


class TestGraphBatch:

    @pytest.mark.parametrize('n', [1, 3, 10, 1000])
    def test_from_data_list(self, n):
        datalist = [random_data(5, 3, 4) for _ in range(n)]
        batch = GraphBatch.from_data_list(datalist)
        assert batch.x.shape[0] > n
        assert batch.e.shape[0] > n
        assert batch.g.shape[0] == n
        assert batch.x.shape[1] == 5
        assert batch.e.shape[1] == 3
        assert batch.g.shape[1] == 4

    def test_to_nx(self):
        datalist = [random_data(5, 3, 4) for _ in range(1000)]
        batch = GraphBatch.from_data_list(datalist)
        batch.to_networkx_list()
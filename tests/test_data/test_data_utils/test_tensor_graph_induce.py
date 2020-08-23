from caldera.data.utils import tensor_induce
import pytest
from caldera.data import GraphBatch
import torch


@pytest.mark.parametrize(
    "random_data", [(GraphBatch, None, (1000, 5, 4, 3))], indirect=True
)
def test_tensor_induce(random_data):

    nodes = torch.LongTensor([[0]])
    tensor_induce(random_data, nodes, 1)

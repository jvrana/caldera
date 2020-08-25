import pytest
import torch

from caldera.data.utils import bfs_nodes
from caldera.utils import deterministic_seed


@pytest.mark.parametrize(
    "src",
    [0, 1, (0, 1), [0, 1], torch.LongTensor([0, 1, 2])],
    ids=lambda x: "src='{}'".format(str(x)),
)
@pytest.mark.parametrize("d", [0, 1, 3, None], ids=lambda x: "depth={}".format(x))
def test_bfs_edges_call_signature(src, d):
    deterministic_seed(0)
    edges = torch.randint(20, (2, 1000))

    nodes = bfs_nodes(src, edges, depth=d)
    if d is None:
        assert len(nodes) == len(torch.unique(edges))
    elif d == 0:
        assert len(nodes) == 0
    print(nodes)

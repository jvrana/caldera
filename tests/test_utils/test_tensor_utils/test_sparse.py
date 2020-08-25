import pytest
import torch

from caldera.utils.sparse import scatter_coo
from caldera.utils.sparse import scatter_coo_fill


@pytest.mark.parametrize(
    ("n", "m", "o"), [(2, 10, 3), (3, 10, 3), (1, 1, 1), (2, 10, 0), (0, 10, 3)]
)
def test_scatter_coo(n, m, o):
    s1 = (n, m)
    if n is None:
        s1 = (m,)
    s2 = (m, o)
    if o is None:
        s2 = (m,)
    indices = torch.randint(1, 10, s1)
    values = torch.randn(s2)
    scatter_coo(indices, values)


@pytest.mark.parametrize(
    ("n", "m", "o"),
    [(None, 10, None), (1, 10, None), (3, 10, None), (None, 10, 1), (None, 10, 3)],
)
def test_scatter_coo_1dim(n, m, o):
    s1 = (n, m)
    if n is None:
        s1 = (m,)
    s2 = (m, o)
    if o is None:
        s2 = (m,)
    indices = torch.randint(1, 10, s1)
    values = torch.randn(s2)
    scatter_coo(indices, values)


@pytest.mark.parametrize(("n", "m"), [(2, 10)])
@pytest.mark.parametrize("values", [0, torch.tensor(0), torch.tensor([0, 1, 2])])
def test_scatter_fill(n, m, values):
    s1 = (n, m)
    if n is None:
        s1 = (m,)

    indices = torch.randint(1, 10, s1)
    scatter_coo_fill(indices, values)

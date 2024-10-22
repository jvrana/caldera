import pytest
import torch

from caldera.utils import same_storage

view_methods = {
    "slice_:10": lambda x: x[:10],
    "slice_10:": lambda x: x[10:],
    "slice_5:10": lambda x: x[5:10],
    "cpu": lambda x: x.cpu(),
    "contiguous": lambda x: x.contiguous(),
}

copy_methods = {
    "clone": lambda x: x.clone(),
    "torch.tensor": lambda x: torch.tensor(x),
    "to(torch.float64)": lambda x: x.to(torch.float64),
}

if torch.cuda.is_available():
    device = "cuda:" + str(torch.cuda.current_device())
    copy_methods["to(" + device + ")"] = lambda x: x.to(device)


def parameterize(n, d):
    args = []
    ids = []
    for k, v in d.items():
        args.append(v)
        ids.append(k)
    return pytest.mark.parametrize(n, args, ids=ids)


@pytest.fixture(
    params=[
        torch.randn(100),
        torch.randn((10, 9)),
        torch.randn((2, 3)),
        torch.randn((2, 0)),
        torch.randn((0, 2)),
        torch.tensor([]),
    ]
)
def example(request):
    return request.param


@parameterize("f", view_methods)
def test_same_storage_view_methods(f, example):
    a = example
    b = f(a)
    assert same_storage(a, b, empty_does_not_share_storage=False)
    assert same_storage(b, a, empty_does_not_share_storage=False)


@parameterize("f", copy_methods)
def test_same_storage_copy_methods(f, example):
    a = example
    b = f(a)
    assert not same_storage(a, b)
    assert not same_storage(b, a)

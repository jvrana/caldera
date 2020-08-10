from pyrographnets.utils import same_storage
import torch
import pytest

view_methods = {
    "slice": lambda x: x[:10],
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


@parameterize("f", view_methods)
@pytest.mark.parametrize('a', [
    torch.randn(100),
    torch.randn((10, 9)),
    torch.randn((2, 3))
])
def test_same_storage_view_methods(f, a):
    b = f(a)
    assert same_storage(a, b)
    assert same_storage(b, a)


@parameterize("f", copy_methods)
@pytest.mark.parametrize('a', [
    torch.randn(100),
    torch.randn((10, 9)),
    torch.randn((2, 3))
])
def test_same_storage_copy_methods(f, a):
    b = f(a)
    assert not same_storage(a, b)
    assert not same_storage(b, a)



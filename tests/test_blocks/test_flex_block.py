import pytest
import torch

from caldera.gnn.blocks import Flex


def test_flex_block():
    flex_linear = Flex(torch.nn.Linear)
    model = flex_linear(Flex.d(), 11)
    print(model.__str__())
    print(model.__repr__())
    x = torch.randn((30, 55))
    model(x)
    print(model.__str__())
    print(model.__repr__())


@pytest.mark.parametrize("x", [16, 32, 44])
def test_flex_block_chain(x):

    model = torch.nn.Sequential(
        Flex(torch.nn.Linear)(Flex.d(), 16),
        Flex(torch.nn.Linear)(Flex.d(), 32),
        Flex(torch.nn.Linear)(Flex.d(), 64),
    )

    data = torch.randn((10, x))
    out = model(data)
    assert out.shape[1] == 64


@pytest.mark.parametrize("x", [16, 32, 44])
def test_flex_block_chain_syntatic_sugar(x):

    model = torch.nn.Sequential(
        Flex(torch.nn.Linear)(..., 16),
        Flex(torch.nn.Linear)(..., 32),
        Flex(torch.nn.Linear)(..., 64),
    )

    data = torch.randn((10, x))
    out = model(data)
    assert out.shape[1] == 64


@pytest.mark.parametrize("x", [16, 32, 44])
def test_flex_block_chain_syntatic_sugar_two_ellipsis(x):

    model = torch.nn.Sequential(
        Flex(torch.nn.Linear)(..., Flex.d(0)),
    )

    data = torch.randn((10, x))
    out = model(data)
    print(out)
    print(list(model.parameters()))

@pytest.mark.parametrize("x", [16, 32, 44])
def test_flex_encoder(x):

    Dim = Flex.d(0)
    FlexLin = Flex(torch.nn.Linear)
    model = torch.nn.Sequential(
        FlexLin(Dim, 10),
        FlexLin(..., 8),
        FlexLin(..., 8),
        FlexLin(..., Dim)
    )

    data = torch.randn((10, x))
    out = model(data)
    assert out.shape[1] == x


def test_print_flex_block():
    m = Flex(torch.nn.Linear)
    s = str(m)
    r = repr(m)
    print(s)
    print(r)
    assert "Linear" in s
    assert "Linear" in r


def test_print_flex_block_init():
    m = Flex(torch.nn.Linear)(Flex.d(), 4)
    s = str(m)
    r = repr(m)
    print(s)
    print(r)
    assert "Linear" in s
    assert "Linear" in r


def test_print_flex_block_init_sugar():
    m = Flex(torch.nn.Linear)(..., 4)
    s = str(m)
    r = repr(m)
    print(s)
    print(r)
    assert "Linear" in s
    assert "Linear" in r


@pytest.mark.parametrize("x", [16, 32, 44])
def test_flex_block_custom_position(x):
    class FooBlock(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.block = torch.nn.Linear(a, b)

        def forward(self, steps, data):
            return self.block(data)

    model = Flex(FooBlock)(Flex.d(1), 16)
    data = torch.randn((10, x))
    model("arg0", data)


@pytest.mark.parametrize("dtype", [None, torch.float64])
@pytest.mark.parametrize("dtype2", [None, torch.float32])
def test_flex_block_to_playback(dtype, dtype2):
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = Flex(torch.nn.Linear)(Flex.d(), 5)

        def forward(self, x):
            return self.layers(x)

    net = Network()
    if dtype:
        net.to(torch.float64)

    print("Before")
    print(list(net.state_dict()))

    if not dtype:
        data = torch.randn(5, 8)
    else:
        data = torch.randn(5, 8, dtype=dtype)
    net(data)

    print("After")
    print(list(net.state_dict()))
    for p in net.parameters():
        if dtype:
            assert p.dtype is dtype

    if dtype2:
        net.to(dtype2)
    for p in net.parameters():
        if dtype2:
            assert p.dtype is dtype2


class TestFlexDocString:
    def test_module_doc_string(self):
        f = Flex(torch.nn.Linear)
        assert torch.nn.Linear.__doc__ in f.__call__.__doc__

    def test_call(self):
        f = Flex(torch.nn.Linear)(10, 10)
        print(torch.nn.Linear(10, 10).__call__.__str__())
        print(f.forward.__doc__)
        print(f.__call__.__doc__)

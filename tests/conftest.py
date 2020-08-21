from os.path import abspath
from os.path import dirname
from os.path import join

import pytest
import torch

from caldera.utils.tensorboard import new_writer as new_summary_writer
from caldera.utils import deterministic_seed
from caldera.data import GraphData, GraphBatch


runs = join(abspath(dirname(__file__)), "pytest_runs")


def pytest_addoption(parser):
    parser.addoption(
        "--cuda", action="store", default=False, help="option: true or false"
    )


@pytest.fixture
def allow_cuda(request):
    return request.config.getoption("--cuda")


@pytest.fixture(scope="module")
def new_writer():
    """Return a function that creates a new writer.

    .. code-block:: python

        def test_writer(new_writer):
            writer = new_writer('my_directory', suffix='_test')
    :return:
    """

    def make_new_writer(directory, *args, **kwargs):
        return new_summary_writer(join(runs, directory), *args, **kwargs)

    return make_new_writer


def get_cuda_device():
    if torch.cuda.is_available():
        return "cuda:" + str(torch.cuda.current_device())


devices = ["cpu"]
if get_cuda_device():
    devices.append(get_cuda_device())


@pytest.fixture(params=devices)
def device(request, allow_cuda):
    device = request.param
    if not allow_cuda and "cuda" in request.param:
        pytest.skip("--cuda=False")
    return device


@pytest.fixture(params=list(range(3)), ids=lambda x: "seed" + str(x))
def seeds(request):
    """
    Example usage of fixture

    .. code-block:: python

        @pytest.mark.parametrize("seeds", list(range(10)), ids=lambda x: "seed" + str(x), indirect=True)
        def test_foo(seeds):
            pass # do stuff


    :param request:
    :return:
    """
    deterministic_seed(request.param)


@pytest.fixture(params=[GraphData, GraphBatch])
def random_data(request):
    default_args = (5, 4, 3)
    default_batch_size = 100
    default_kwargs = {}

    args = None
    kwargs = None
    if isinstance(request.param, tuple):
        if len(request.param) == 2:
            data_cls, seed = request.param
        elif len(request.param) == 3:
            data_cls, seed, args = request.param
            kwargs = {}
        else:
            data_cls, seed, args, kwargs = request.param
        if seed is not None:
            deterministic_seed(seed)
    else:
        data_cls = request.param
        deterministic_seed(0)

    if args is None:
        if data_cls is GraphBatch:
            args = (default_batch_size,) + default_args
        else:
            args = default_args
    if kwargs is None:
        kwargs = default_kwargs

    if hasattr(data_cls, GraphBatch.random_batch.__name__):
        return data_cls.random_batch(*args, **kwargs)
    else:
        return data_cls.random(*args, **kwargs)

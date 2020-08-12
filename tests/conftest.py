from os.path import abspath
from os.path import dirname
from os.path import join

import pytest
import torch

from pyrographnets.utils.tensorboard import new_writer as new_summary_writer

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

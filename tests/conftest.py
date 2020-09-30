from functools import partial
from os.path import abspath
from os.path import dirname
from os.path import join
from typing import Dict
from typing import Tuple

import pytest
import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils import deterministic_seed
from caldera.utils.tensorboard import new_writer as new_summary_writer


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "env(name): mark test to run only on named environment"
    )
    config.addinivalue_line("markers", "slowtest: marks a long running test")
    config.addinivalue_line(
        "markers", "train: marks a test that performs network training (typically slow)"
    )


#############################
# Incremental Testing
#############################

"""
Use `pytest.mark.incremental` on pytest Class definition to do incremental tests.
"""

# store history of failures per test class name and per index in parametrize (if parametrize used)
_test_failed_incremental: Dict[str, Dict[Tuple[int, ...], str]] = {}


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        # incremental marker is used
        if call.excinfo is not None:
            # the test has failed
            # retrieve the class name of the test
            cls_name = str(item.cls)
            # retrieve the index of the test (if parametrize is used in combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the test function
            test_name = item.originalname or item.name
            # store in _test_failed_incremental the original name of the failed test
            _test_failed_incremental.setdefault(cls_name, {}).setdefault(
                parametrize_index, test_name
            )


def _pytest_env_mark_setup(item):
    envnames = [mark.args[0] for mark in item.iter_markers(name="env")]
    env = item.config.getoption("-E")
    provided_but_env_item_missing = env and env not in envnames
    not_provided_but_env_item_present = not env and envnames
    if provided_but_env_item_missing or not_provided_but_env_item_present:
        pytest.skip("test requires env in {!r}".format(envnames))


def _pytest_auto_mark_benchmark(item):
    """Automatically mark tests that use the `benchmark` fixture."""
    marks = [mark for mark in item.iter_markers(name="benchmark")]
    if not marks and "benchmark" in item.fixturenames:
        item.add_marker(pytest.mark.benchmark)


def pytest_runtest_setup(item):
    _pytest_auto_mark_benchmark(item)
    _pytest_env_mark_setup(item)
    _pytest_incr_mark_setup(item)


def _pytest_incr_mark_setup(item):
    if "incremental" in item.keywords:
        # retrieve the class name of the test
        cls_name = str(item.cls)
        # check if a previous test has failed for this class
        if cls_name in _test_failed_incremental:
            # retrieve the index of the test (if parametrize is used in combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the first test function to fail for this class name and index
            test_name = _test_failed_incremental[cls_name].get(parametrize_index, None)
            # if name found, test has failed for the combination of class name & test name
            if test_name is not None:
                pytest.xfail("previous test failed ({})".format(test_name))


#############################
# END
# Incremental Testing
#############################


#############################
# CUDA Device Param Setup
#############################

runs = join(abspath(dirname(__file__)), "pytest_runs")


def pytest_addoption(parser):
    parser.addoption(
        "--cuda", action="store", default=False, help="option: true or false"
    )
    parser.addoption(
        "-E",
        action="store",
        metavar="NAME",
        help="only run tests matching the environment NAME.",
    )


@pytest.fixture
def allow_cuda(request):
    return request.config.getoption("--cuda")


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


#############################
# END
# CUDA Device Param Setup
#############################


@pytest.fixture(scope="module")
def new_writer():
    """Return a function that creates a new writer.

    .. code-block::

        def test_writer(new_writer):
            writer = new_writer('my_directory', suffix='_test')
    :return:
    """

    def make_new_writer(directory, *args, **kwargs):
        return new_summary_writer(join(runs, directory), *args, **kwargs)

    return make_new_writer


@pytest.fixture(params=list(range(3)), ids=lambda x: "seed" + str(x))
def seeds(request):
    """Example usage of fixture.

    .. code-block::

        @pytest.mark.parametrize("seeds", list(range(10)), ids=lambda x: "seed" + str(x), indirect=True)
        def test_foo(seeds):
            pass # do stuff


    :param request:
    :return:
    """
    deterministic_seed(request.param)


#############################
# Data fixtures
#############################


def create_data_constructor(param):
    default_args = (5, 4, 3)
    default_batch_size = 100
    default_kwargs = {}

    args = None
    kwargs = None
    if isinstance(param, tuple):
        if len(param) == 2:
            data_cls, seed = param
        elif len(param) == 3:
            data_cls, seed, args = param
            kwargs = {}
        else:
            data_cls, seed, args, kwargs = param
        if seed is not None:
            deterministic_seed(seed)
    else:
        data_cls = param
        deterministic_seed(0)

    if args is None:
        if data_cls is GraphBatch:
            args = (default_batch_size,) + default_args
        else:
            args = default_args
    if kwargs is None:
        kwargs = default_kwargs

    if hasattr(data_cls, GraphBatch.random_batch.__name__):
        rndm_func = data_cls.random_batch
    else:
        rndm_func = data_cls.random
    return partial(rndm_func, *args, **kwargs)


@pytest.fixture(params=[GraphData, GraphBatch])
def random_data_constructor(request):
    return create_data_constructor(request.param)


@pytest.fixture(params=[GraphData, GraphBatch])
def random_data(request):
    return create_data_constructor(request.param)()


@pytest.fixture(params=[(100, GraphData)])
def random_data_list(request):
    datalist = []
    num = request.param[0]
    params = tuple(list(request.param)[1:])
    if len(params) == 1:
        params = params[0]
    for i in range(num):
        datalist.append(create_data_constructor(params)())
    return datalist


#############################
# END
# Data fixtures
#############################


#############################
# Test fixtures
#############################


@pytest.mark.incremental
class TestConftestFixtures:
    def test_random_data(self, random_data):
        assert issubclass(random_data.__class__, GraphData)

    @pytest.mark.parametrize("random_data", [GraphData], indirect=True)
    def test_random_data_indirect(self, random_data):
        assert issubclass(random_data.__class__, GraphData)

    @pytest.mark.parametrize("random_data", [GraphData, GraphBatch], indirect=True)
    def test_random_data_indirect2(self, random_data):
        print(random_data)

    @pytest.mark.parametrize(
        "random_data_constructor", [GraphData, GraphBatch], indirect=True
    )
    def test_data_constructor(self, random_data_constructor):
        assert callable(random_data_constructor)
        assert random_data_constructor()

    def test_init_dataset(self, random_data_list):
        pass

    @pytest.mark.parametrize("random_data_list", [(100, GraphData)], indirect=True)
    def test_init_dataset_indirect(self, random_data_list):
        random_data_list


#############################
# END
# Data fixtures
#############################

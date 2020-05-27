from os.path import abspath
from os.path import dirname
from os.path import isdir
from os.path import join

import pytest

from pyro_graph_nets.utils.tensorboard import new_writer as new_summary_writer

runs = join(abspath(dirname(__file__)), "..", ".runs")


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

import os
import sys

import pytest


@pytest.fixture
def temp_sys_path(request):
    assert request.param.__class__ is str
    assert os.path.isdir(request.param)
    sys.path.insert(0, request.param)
    yield
    sys.path.remove(request.param)

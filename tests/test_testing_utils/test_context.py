from pyrographnets.utils.testing import pytest_contexts, _context_manager_test_cases
import pytest


# @pytest.mark.parametrize("x", [1, 2, 3])
# @pytest_contexts("mycases", ["is odd", "is even"], mode="accumulate")
# def test_context(x, mycases):
#     with mycases:
#         assert x % 2 == 1
#
#     with mycases:
#         assert x % 2 == 0

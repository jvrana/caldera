# from caldera.utils import torch_unique
# import torch
# import pytest
#
#
# @pytest.mark.increment
# class TestUnique:
#     @pytest.mark.parametrize("n", [1, 10, 100])
#     @pytest.mark.parametrize(
#         "kwargs",
#         [
#             {"return_counts": False, "return_inverse": False},
#             {"return_counts": True, "return_inverse": False},
#             {"return_counts": False, "return_inverse": True},
#             {"return_counts": True, "return_inverse": True},
#         ],
#         ids=lambda x: str(x),
#     )
#     def test_torch_unique_1d(self, n, kwargs):
#         a = torch.randint(10, (n,))
#         b, inverse, counts = torch_unique(a, **kwargs)
#         expected = torch.unique(a, **kwargs)
#         if kwargs["return_counts"] and kwargs["return_inverse"]:
#             expected, expected_idx, expected_counts = expected
#         elif kwargs["return_counts"]:
#             expected, expected_counts = expected
#         elif kwargs["return_inverse"]:
#             expected, expected_idx = expected
#
#         assert torch.all(b == expected)
#         if not kwargs["return_counts"]:
#             assert counts.shape[0] == 0
#         else:
#             assert torch.all(counts == expected_counts)
#
#         if not kwargs["return_inverse"]:
#             assert inverse.shape[0] == 0
#         else:
#             assert torch.all(inverse == expected_idx)

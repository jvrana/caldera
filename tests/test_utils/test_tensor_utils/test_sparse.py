import pytest
import torch

from caldera.utils.indexing import unroll_index
from caldera.utils.sparse import scatter_coo


@pytest.mark.parametrize(
    "shape",
    [(2, 3), torch.Size([2, 3]), (2,), (1, 2, 3), (5, 3, 4, 2)],
    ids=lambda x: str(x),
)
def test_unroll_indices(shape):
    arr = torch.randn(shape)
    idxs = unroll_index(shape)
    assert torch.all(arr[idxs] == arr.flatten())


class TestScatterCoo:
    def validate_scatter_coo(self, indices, values, size=None):
        if size is None:
            expected_size = tuple((indices.max(dim=1).values + 1).tolist())
        elif size[-1] is ...:
            expected_size = size[:-1] + values.shape[1:]
        else:
            expected_size = size
        expected = torch.zeros(expected_size)
        expected[indices.unbind()] = values

        result = scatter_coo(indices, values, size)

        if expected_size is not None:
            assert result.size() == expected_size
        assert torch.all(result.to_dense() == expected)

    def test_scatter_coo_2D(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.validate_scatter_coo(indices, values, torch.Size([4, 5]))

    def test_scatter_coo_size_2D_size(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.validate_scatter_coo(indices, values, torch.Size([10, 10]))

    def test_scatter_coo_size_2D_size_ellipsis(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.validate_scatter_coo(indices, values, size=(10, 10, ...))

    def test_scatter_coo_3D(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        self.validate_scatter_coo(indices, values, size=torch.Size([4, 5, 2]))

    def test_scatter_coo_3D(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        self.validate_scatter_coo(indices, values, size=(10, 10, ...))

    @pytest.mark.xfail(
        strict=True, raises=RuntimeError, reason="Size has wrong dimensions"
    )
    def test_scatter_coo_3D_fail(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        self.validate_scatter_coo(indices, values, size=(10, 10))

    @pytest.mark.xfail(
        strict=True, raises=RuntimeError, reason="Tensors have wrong dimensions"
    )
    def test_scatter_coo_size_mismatch(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([[0, 1, 3]])
        self.validate_scatter_coo(indices, values)

    def test_scatter_coo_infer_second_dim(self):
        indices = torch.LongTensor([0, 1, 2, 3])
        values = torch.tensor([10, 20, 30, 40])
        matrix = scatter_coo(indices, values)
        assert matrix.size() == torch.Size([1, 4])

    def test_scatter_coo_5D(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1], [1, 2, 3, 4]])
        values = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
        self.validate_scatter_coo(indices, values, size=(10, 10, 10, ...))


class TestScatterCooExpand:
    def validate_scatter_coo(self, indices, values, size=None):
        if torch.is_tensor(values):
            value_shape = values.shape
        else:
            value_shape = torch.Size([])

        if size is None:
            expected_size = tuple((indices.max(dim=1).values + 1).tolist())
        elif size[-1] is ...:
            expected_size = size[:-1] + value_shape
        else:
            expected_size = size
        expected = torch.zeros(expected_size)
        expected[indices.to(torch.long).unbind()] = values

        result = scatter_coo(indices, values, size=size, expand=True)

        if expected_size is not None:
            assert result.size() == expected_size
        assert torch.all(result.to_dense() == expected)

    def test_scatter_coo_2D_expand_with_num(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = 1
        self.validate_scatter_coo(indices, values, torch.Size([4, 5]))

    def test_scatter_coo_2D_expand_with_0D_tensor(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor(1, dtype=torch.float)
        self.validate_scatter_coo(indices, values, torch.Size([4, 5]))

    def test_scatter_coo_2D_expand_with_1D_tensor(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([1], dtype=torch.float)
        self.validate_scatter_coo(indices, values, torch.Size([4, 5, 1]))

    def test_scatter_coo_2D_expand_with_1D_tensor_ellipsis(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([1], dtype=torch.float)
        self.validate_scatter_coo(indices, values, (4, 5, ...))

    def test_scatter_coo_2D_expand_with_1D_tensor2(self):
        indices = torch.Tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([1, 2, 3], dtype=torch.float)
        self.validate_scatter_coo(indices, values, torch.Size([4, 5, 3]))

    def test_scatter_coo_2D_expand_with_1D_tensor2_ellipsis(self):
        indices = torch.Tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor([1, 2, 3], dtype=torch.float)
        self.validate_scatter_coo(indices, values, (4, 5, ...))

    def test_scatter_coo_2D_expand_with_2D_tensor(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1]])
        values = torch.tensor(
            [[1, 2, 3, 1], [3, 4, 5, 1], [3, 4, 5, 5]], dtype=torch.float
        )
        self.validate_scatter_coo(indices, values, (4, 5, ...))

    def test_scatter_coo_3D_expand_with_2D_tensor(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1], [3, 4, 5, 10]])
        values = torch.tensor(
            [[1, 2, 3, 1], [3, 4, 5, 1], [3, 4, 5, 5]], dtype=torch.float
        )
        self.validate_scatter_coo(indices, values, (50, 50, 50, ...))

    @pytest.mark.xfail(
        raises=ValueError, strict=True, reason="n dims does not match index dims"
    )
    def test_scatter_expand_fail(self):
        indices = torch.LongTensor([[0, 1, 2, 3], [4, 3, 2, 1], [3, 4, 5, 10]])
        values = torch.tensor(
            [[1, 2, 3, 1], [3, 4, 5, 1], [3, 4, 5, 5]], dtype=torch.float
        )
        scatter_coo(indices, values, expand=True, size=(50, 50, ...))

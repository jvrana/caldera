import torch


def same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Checks if two tensors share storage."""
    print(x)
    print(y)
    print(x.shape)
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)

import torch

def same_storage(x: torch.tensor, y: torch.tensor) -> bool:
    """Check if two tensors have the same underlying storage"""
	x_ptrs = set(e.data_ptr() for e in x.view(-1))
	y_ptrs = set(e.data_ptr() for e in y.view(-1))
	return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)
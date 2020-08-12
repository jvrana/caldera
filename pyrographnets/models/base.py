import torch


class GraphNetworkBase(torch.nn.Module):
    """Base class for GraphNetwork modules."""

    # def reset_parameters(self):
    #     for child in self.children():
    #         if hasattr(child, 'reset_parameters'):
    #             child.reset_parameters()

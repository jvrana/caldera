from torch import nn
import torch

# try:
#     from rich.console import print as logprint
# except ImportError:
#     logprint = print
#
# try:
#     from rich.console import Console
#     console = Console()
#     def logprint(*args, **kwargs):
#         console.print(*args, **kwargs)
# except ImportError:
#     logprint = print


def check_back_prop(model: nn.Module, out: torch.Tensor):
    """Check the backpropogation for a module"""
    loss = nn.MSELoss()(out, torch.rand_like(out))
    grads = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad = p.grad.detach().clone()
        else:
            grad = None
        grads[n] = (grad,)

    loss.backward(retain_graph=True)
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad = p.grad.detach().clone()
        else:
            grad = None
        grads[n] += (grad,)

    diff = {}
    for n in grads:
        p1, p2 = grads[n]
        no_change = False
        gradient = True
        if p2 is None or p2.sum() == 0.0:
            x = no_change
        else:
            x = gradient
        diff[n] = x
    return diff

from torch import nn

from caldera import gnn
from caldera.data import GraphBatch


def check_gradients(model, loss):
    grads = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad = p.grad.detach().clone()
        else:
            grad = None
        grads[n] = (grad,)

    loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad = p.grad.detach().clone()
        else:
            grad = None
        grads[n] += (grad,)

    diff = {}
    for n in grads:
        p1, p2 = grads[n]
        no_change = "[red]no change[/red]"
        gradient = "[green]gradient[/green]"
        if p2 is None or p2.sum() == 0.0:
            x = no_change
        else:
            x = gradient
        diff[n] = x
    return diff

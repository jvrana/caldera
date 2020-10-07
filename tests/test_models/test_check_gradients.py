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


def test_():
    data = GraphBatch.random_batch(10, 5, 4, 3)
    target = data.detach().copy().randomize_(1, 1, 1)

    core = gnn.GraphCore(
        edge_block=gnn.AggregatingEdgeBlock(gnn.Flex(gnn.Dense)(..., 1)),
        node_block=gnn.AggregatingNodeBlock(
            gnn.Flex(gnn.Dense)(..., 1), gnn.Aggregator("add")
        ),
        global_block=gnn.AggregatingGlobalBlock(
            gnn.Flex(gnn.Dense)(..., 1), gnn.Aggregator("add"), gnn.Aggregator("add")
        ),
        pass_global_to_edge=True,
        pass_global_to_node=True,
    )

    core(data)

    out = core(data)
    loss = nn.MSELoss()(out.g, target.g)

    from rich.console import Console

    console = Console()
    for k, v in check_gradients(core, loss).items():
        console.print(k)
        console.print(v)
        print()

    core.global_block.block_dict["edge_aggregator"]

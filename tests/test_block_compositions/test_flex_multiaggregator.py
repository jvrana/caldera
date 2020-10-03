import torch

from caldera.gnn.blocks import AggregatingNodeBlock
from caldera.gnn.blocks import Dense
from caldera.gnn.blocks import Flex
from caldera.gnn.blocks import MultiAggregator


def test_flexible_multiaggregator():
    net = Flex(MultiAggregator)(Flex.d(), aggregators=["add"])
    data = torch.randn((10, 5), dtype=torch.float)
    idx = torch.randint(0, 2, (10,), dtype=torch.long)
    net(data, idx, dim=0, dim_size=20)
    print(net)


def test_flexible_agg_node_block_mult_agg():
    net = AggregatingNodeBlock(
        Flex(Dense)(Flex.d(), 25),
        edge_aggregator=Flex(MultiAggregator)(Flex.d(), aggregators=["add"]),
    )
    n_edges = 10
    e_feat = 5
    n_nodes = 20
    n_feat = 7
    edata = torch.randn((n_edges, e_feat), dtype=torch.float)
    ndata = torch.randn((n_nodes, n_feat), dtype=torch.float)
    edges = torch.randint(0, 2, (2, n_edges), dtype=torch.long)
    net(node_attr=ndata, edge_attr=edata, edges=edges)


def test_():
    net = AggregatingNodeBlock(
        Flex(Dense)(Flex.d(), 25),
        edge_aggregator=Flex(MultiAggregator)(Flex.d(), aggregators=["add"]),
    )
    n_edges = 10
    e_feat = 5
    n_nodes = 20
    n_feat = 7
    edata = torch.randn((n_edges, e_feat), dtype=torch.float)
    ndata = torch.randn((n_nodes, n_feat), dtype=torch.float)
    edges = torch.randint(0, 2, (2, n_edges), dtype=torch.long)
    out = net(node_attr=ndata, edge_attr=edata, edges=edges)

    print(out)
    var = out
    output_nodes = (
        (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)
    )
    print(output_nodes)
    next = output_nodes[0].next_functions


def make_dot(var, params=None):
    """Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    output_nodes = (
        (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)
    )

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
            elif hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "{}\n {}".format(name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor="lightblue")
            elif var in output_nodes:
                dot.node(
                    str(id(var)), str(type(var).__name__), fillcolor="darkolivegreen1"
                )
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

        # handle multiple outputs
        if isinstance(var, tuple):
            for v in var:
                add_nodes(v.grad_fn)
        else:
            add_nodes(var.grad_fn)

        # resize_graph(dot)

        return dot


def add_nodes(var):
    seen = []
    if var not in seen:
        if torch.is_tensor(var):
            # note: this used to show .saved_tensors in pytorch0.2, but stopped
            # working as it was moved to ATen and Variable-Tensor merged
            dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
        elif hasattr(var, "variable"):
            u = var.variable
            name = param_map[id(u)] if params is not None else ""
            node_name = "{}\n {}".format(name, size_to_str(u.size()))
            dot.node(str(id(var)), node_name, fillcolor="lightblue")
        elif var in output_nodes:
            dot.node(str(id(var)), str(type(var).__name__), fillcolor="darkolivegreen1")
        else:
            dot.node(str(id(var)), str(type(var).__name__))
        seen.add(var)
        if hasattr(var, "next_functions"):
            for u in var.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
        if hasattr(var, "saved_tensors"):
            for t in var.saved_tensors:
                dot.edge(str(id(t)), str(id(var)))
                add_nodes(t)

        # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    # resize_graph(dot)

    return dot

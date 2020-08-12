import torch

from caldera.blocks import AggregatingNodeBlock
from caldera.blocks import Flex
from caldera.blocks import MLP
from caldera.blocks import MultiAggregator


def test_flexible_multiaggregator():
    net = Flex(MultiAggregator)(Flex.d(), aggregators=["add"])
    data = torch.randn((10, 5), dtype=torch.float)
    idx = torch.randint(0, 2, (10,), dtype=torch.long)
    net(data, idx, dim=0, dim_size=20)
    print(net)


def test_flexible_agg_node_block_mult_agg():
    net = AggregatingNodeBlock(
        Flex(MLP)(Flex.d(), 25),
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

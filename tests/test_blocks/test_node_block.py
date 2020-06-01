from pyrographnets.blocks import MLP, NodeBlock, AggregatingNodeBlock, Aggregator
import torch


def test_init_node_block():
    # test NodeBlock
    node_encoder = NodeBlock(MLP(5, 16, 10))

    node_attr = torch.randn(10, 5)

    for p in node_encoder.parameters():
        assert p.requires_grad

def test_init_agg_node_block():
    # test AggregatingNodeBlock

    node_model = AggregatingNodeBlock(MLP(5, 16, 10), Aggregator('mean'))

    edge_attr = torch.randn(20, 3)
    edges = torch.randint(0, 40, torch.Size([2, 20]))
    node_attr = torch.randn(40, 2)

    node_model(node_attr, edge_attr, edges).shape

    for p in node_model.parameters():
        assert p.requires_grad
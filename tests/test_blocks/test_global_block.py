from pyrographnets.blocks import MLP, GlobalBlock, AggregatingGlobalBlock, Aggregator
import torch


def test_init_global_block():
    # test GlobalBlock
    global_encoder = GlobalBlock(MLP(3, 10))

    for p in global_encoder.parameters():
        print(p)
        print(p.requires_grad)

    global_attr = torch.randn(10, 3)
    global_encoder(global_attr).shape

    for p in global_encoder.parameters():
        assert p.requires_grad


def test_init_agg_global_block():
    # test AggregatingGlobalBlock
    global_attr = torch.randn(10, 3)
    edge_attr = torch.randn(20, 3)
    edges = torch.randint(0, 40, torch.Size([2, 20]))
    node_attr = torch.randn(40, 2)
    node_idx = torch.randint(0, 3, torch.Size([40]))
    edge_idx = torch.randint(0, 3, torch.Size([20]))

    global_model = AggregatingGlobalBlock(MLP(8, 16, 10), Aggregator('mean'), Aggregator('mean'))
    out = global_model(global_attr, node_attr, edge_attr, edges, node_idx, edge_idx)

    for p in global_model.parameters():
        assert p.requires_grad

    print(list(global_model.parameters()))
from pyrographnets.blocks import (
    GlobalBlock,
    Aggregator,
    AggregatingGlobalBlock,
    Flex,
    MLP,
)
from pyrographnets.data.utils import random_data
from pyrographnets.data import GraphDataLoader
import torch


def test_train(new_writer):

    # new writer
    writer = new_writer("runs/global_block")

    # initialize random data
    test_data = [random_data(5, 4, 3) for _ in range(1000)]
    test_loader = GraphDataLoader(test_data, batch_size=32, shuffle=True)
    test_mask = (
        torch.tensor([True, True, True, True, False]),
        torch.tensor([True, True, False, False]),
        torch.tensor([True, True, False]),
    )

    # model
    # model = GlobalBlock(MLP(2, 16, 1))
    model = GlobalBlock(Flex(MLP)(Flex.d(), 16, 1))
    model = AggregatingGlobalBlock(
        Flex(MLP)(Flex.d(), 16, 1), Aggregator("add"), Aggregator("add")
    )

    # global_attr, node_attr, edge_attr, edges, node_idx, edge_idx):

    for batch in test_loader:
        batch = batch.mask(*test_mask)
    model(batch.g, batch.x, batch.e, batch.edges, batch.node_idx, batch.edge_idx)

    # model = torch.nn.Sequential(
    #     torch.nn.Linear(2, 16),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(16, 1),
    #     torch.nn.ReLU()
    # )

    # setup initializer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()

    # training
    for epoch in range(10p):

        running_loss = 0.0
        for batch_idx, batch in enumerate(test_loader):
            test_data = batch.mask(*test_mask)
            target_data = batch.mask(*test_mask, invert=True)

            out = model(
                test_data.g,
                test_data.x,
                test_data.e,
                test_data.edges,
                test_data.node_idx,
                test_data.edge_idx,
            )
            loss = loss_fn(out, target_data.g)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        writer.add_scalar("test loss", running_loss, epoch)

        # if epoch % 1 == 0:
        #     k = 'block_dict.mlp.resolved_module.blocks.0.blocks.0.weight'
        #     writer.add_histogram(k, model.state_dict()[k])

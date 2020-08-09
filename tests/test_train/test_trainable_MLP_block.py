from pyrographnets.blocks import Flex, MLP
from pyrographnets.data import GraphData, GraphDataLoader
import torch.optim as optim

def test_train_edge_block():
    epochs = 10
    network = Flex(MLP)(Flex.d(), 16, 1, layer_norm=False)
    input_datalist = [GraphData.random(5, 5, 5) for _ in range(1000)]

    #
    # target_datalist = [GraphData.random(5, 1, 5) for _ in range(1000)]
    # loader = GraphDataLoader((input_datalist, target_datalist), batch_size=100)
    # for batch in loader:
    #     break
    # network(batch.e)
    #
    # for input_batch, target_batch =
    #
    # # zero the parameter gradients
    # optimizer.zero_grad()
    #
    # # forward + backward + optimize
    # outputs = net(inputs)
    # loss = criterion(outputs, labels)
    # loss.backward()
    # optimizer.step()
    #
    # #
    # # for batch in GraphDataLoader()
    # # optimizer.zero_grad()
    # #
    # # # forward + backward + optimize
    # # outputs = network(inputs)
    # # loss = criterion(outputs, labels)
    # # loss.backward()
    # # optimizer.step()
    # #
    # # for e in range(epochs):
    # #
    # #     network.train()
    # #
    # #

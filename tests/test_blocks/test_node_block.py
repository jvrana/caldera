from pyrographnets.blocks import MLP, EdgeBlock, AggregatingEdgeBlock
import torch
# demonstrate independent edge block
edge_encoder = EdgeBlock(MLP(3, 10, 16))

x = torch.randn(20, 3)
out = edge_encoder(x)
print(out.shape)
assert out.shape == torch.Size([20, 16])

for p in edge_encoder.parameters():
    print(p)
    print(p.requires_grad)
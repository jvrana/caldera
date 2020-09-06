import torch
from torch import nn

from caldera.blocks.block import Block
from caldera.data import GraphData


class EdgeBlock(Block):
    def __init__(self, mlp: nn.Module):
        super().__init__({"mlp": mlp}, independent=True)

    def forward(self, edge_attr: torch.tensor):
        results = self.block_dict["mlp"](edge_attr)
        return results

    def forward_from_data(self, data: GraphData):
        return self(data.e)


class AggregatingEdgeBlock(EdgeBlock):
    def __init__(self, mlp: nn.Module):
        super().__init__(mlp)
        self._independent = False

    def forward(
        self,
        *,
        edge_attr: torch.tensor,
        node_attr: torch.tensor,
        edges: torch.tensor,
        global_attr: torch.Tensor = None,
        edge_idx: torch.Tensor = None,
    ):
        to_agg = (node_attr[edges[0]], node_attr[edges[1]])
        if global_attr is not None:
            if edge_idx is None:
                raise RuntimeError(
                    "If `global_attr` provided must also provide `edge_index`"
                )
            to_agg += (global_attr[edge_idx],)
        out = torch.cat([*to_agg, edge_attr], 1)

        return self.block_dict["mlp"](out)

    def forward_from_data(self, data: GraphData):
        return self(data.e, data.x.data.edges)

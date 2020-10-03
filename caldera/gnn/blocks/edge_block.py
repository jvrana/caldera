from typing import Any
from typing import Optional
from typing import Tuple

import torch
from torch import nn

from caldera.data import GraphData
from caldera.gnn.blocks.block import Block


class EdgeBlock(Block):
    def __init__(self, module: nn.Module):
        super().__init__({"module": module}, independent=True)

    def forward(self, edge_attr: torch.FloatTensor):
        results = self.block_dict["module"](edge_attr)
        return results

    def forward_from_data(self, data: GraphData):
        return self(data.e)


class AggregatingEdgeBlock(EdgeBlock):
    def __init__(self, module: nn.Module):
        super().__init__(module)
        self._independent = False

    def forward(
        self,
        *args: Tuple,
        edge_attr: torch.FloatTensor,
        node_attr: torch.FloatTensor,
        edges: torch.LongTensor,
        global_attr: Optional[torch.FloatTensor] = None,
        edge_idx: Optional[torch.LongTensor] = None,
    ) -> Any:
        to_agg = (node_attr[edges[0]], node_attr[edges[1]])
        if global_attr is not None:
            if edge_idx is None:
                raise RuntimeError(
                    "If `global_attr` provided must also provide `edge_index`"
                )
            to_agg += (global_attr[edge_idx],)
        out = torch.cat([*to_agg, edge_attr], 1)

        return self.block_dict["module"](out)

    def forward_from_data(self, data: GraphData):
        return self(data.e, data.x.data.edges)

from functools import partial

import torch
from torch import nn

from pyro_graph_nets.blocks import EdgeBlock
from pyro_graph_nets.blocks import GlobalBlock
from pyro_graph_nets.blocks import NodeBlock
from pyro_graph_nets.utils.graph_tuple import GraphTuple
from pyro_graph_nets.utils.graph_tuple import replace_key


def gt_wrap_replace(func):
    def forward(gt):
        v, e, u = func(
            gt.node_attr,
            gt.edges.T,
            gt.edge_attr,
            gt.global_attr,
            gt.node_indices,
            gt.edge_indices,
        )
        gt = replace_key(gt, {"node_attr": v, "edge_attr": e, "global_attr": u})
        return gt

    return forward


class IdentityModel(torch.nn.Module):
    def __init__(self, pos):
        self.pos = pos

    def forward(self, *args):
        return args[self.pos]


class GraphAbstractModel(nn.Module):
    def __init__(
        self, edge_model: nn.Module, node_model: nn.Module, global_model: nn.Module
    ):
        super().__init__()

        mod_dict = {
            "edge_model": edge_model,
            "node_model": node_model,
            "global_model": global_model,
        }
        mod_dict = {k: v for k, v in mod_dict.items() if v is not None}

        self.blocks = nn.ModuleDict(mod_dict)
        self.reset_parameters()
        self.forward = gt_wrap_replace(self.forward_helper)

    def reset_parameters(self):

        for mod in self.blocks.values():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()


class GraphEncoder(GraphAbstractModel):
    def forward_helper(
        self,
        node_attr,
        connectivity,
        edge_attr=None,
        u=None,
        node_idx=None,
        edge_idx=None,
    ):

        blocks = dict(self.blocks.items())
        edge_model = blocks.get("edge_model", None)
        node_model = blocks.get("node_model", None)
        global_model = blocks.get("global_model", None)

        row, col = connectivity

        if edge_model is not None:
            edge_attr = edge_model(
                node_attr[row], node_attr[col], edge_attr, u, node_idx, edge_idx
            )
        else:
            edge_attr = None

        if node_model is not None:
            node_attr = node_model(
                node_attr, connectivity, edge_attr, u, node_idx, edge_idx
            )
        else:
            node_attr = None

        if global_model is not None:
            global_attr = global_model(
                node_attr, connectivity, edge_attr, u, node_idx, edge_idx
            )
        else:
            global_attr = torch.zeros_like(u)
            global_attr.requires_grad = True

        return node_attr, edge_attr, global_attr


class GraphNetwork(GraphAbstractModel):
    def forward_helper(
        self,
        node_attr,
        edge_index,
        edge_attr=None,
        global_attr=None,
        node_idx=None,
        edge_idx=None,
    ):
        row, col = edge_index

        blocks = dict(self.blocks.items())
        edge_model = blocks.get("edge_model", None)
        node_model = blocks.get("node_model", None)
        global_model = blocks.get("global_model", None)

        if edge_model is not None:
            edge_attr = edge_model(
                node_attr[row], node_attr[col], edge_attr, global_attr, node_idx, edge_idx
            )

        if node_model is not None:
            node_attr = node_model(
                node_attr, edge_index, edge_attr, global_attr, node_idx, edge_idx
            )

        if global_model is not None:
            global_attr = global_model(
                node_attr, edge_index, edge_attr, global_attr, node_idx, edge_idx
            )

        return node_attr, edge_attr, global_attr


# # TODO: why are the concatenations all mixed up???
# class EncoderProcessDecoder(torch.nn.Module):
#     """enc_vertex_shape=(1, 16), core_vertex_shape=(32, 16),
#     dec_vertex_shape=(16, 16), out_vertex_size=2,
#
#     enc_edge_shape=(1, 16), core_edge_shape=(32, 16),
#     dec_edge_shape=(16, 16), out_edge_size=2, device=device,
#     """
#
#     def __init__(
#         self,
#         v_features: int,
#         e_features: int,
#         u_features: int,
#         v_out: int,
#         e_out: int,
#         u_out: int,
#     ):
#         super().__init__()
#
#         # in_e_features = n_features * 2 + e_features
#         # incoming node features = in_e_features
#
#         latent_size = 16
#
#         # edge features are concatenated along with the src and dest node features
#         e_enc_in = e_features + v_features * 2
#
#         # can be anything
#         e_enc_out = e_enc_in
#
#         # aggregate the edge_features + node features
#         v_enc_in = e_enc_out + u_features + v_features
#         v_enc_out = v_enc_in
#
#         # in the core we concatenate the first latent features for some reason??
#         e_core_in = e_enc_out * 2
#         e_core_out = e_enc_out
#         v_core_in = e_core_out + (u_features + v_features) * 2
#         v_core_out = v_enc_out
#
#         e_dec_in = 160
#         e_dec_out = 160
#         v_dec_in = 160
#         v_dec_out = 160
#
#         #         e_encoded_in = e_features + v_features * 2
#         #         e_encoded_out = e_encoded_in
#         #         v_encoded_size =
#
#         self.encoder = GraphEncoder(
#             EdgeBlock(e_enc_in, [latent_size, e_enc_out], independent=True),
#             NodeBlock(v_enc_in, [latent_size, v_enc_out], independent=True),
#             None,
#         )
#
#         self.core = GraphNetwork(
#             EdgeBlock(e_enc_in * 2, [latent_size, e_enc_out], independent=False),
#             NodeBlock(v_enc_in * 2, [latent_size, e_enc_out], independent=False),
#             None,
#         )
#
#         self.decoder = GraphEncoder(
#             EdgeBlock(11, [16, 11], independent=True),
#             NodeBlock(32, [16, 11], independent=True),
#             None,
#         )
#
#         self.output_transform = OutputTransform(
#             torch.nn.Linear(5, 5), torch.nn.Linear(1, 1), None
#         )
#
#     @staticmethod
#     def graph_tuple_to_args(graph_tuple):
#         return (
#             graph_tuple.node_attr,
#             graph_tuple.edges.T,
#             graph_tuple.edge_attr,
#             graph_tuple.global_attr,
#         )
#
#     def forward(self, graph_tuple, steps: int):
#         # TODO: copy and detatch?
#         encoder = gt_wrap_replace(self.encoder)
#         core = gt_wrap_replace(self.core)
#         decoder = gt_wrap_replace(self.decoder)
#         output_transform = gt_wrap_replace(self.output_transform)
#         latent = encoder(graph_tuple)
#         latent0 = latent
#         output_ops = []
#         for _ in range(steps):
#             core_input = cat_gt(*[latent0, latent])
#             latent = core(core_input)
#             decoded_op = output_transform(decoder(latent))
#             output_ops.append(decoded_op)
#         return output_ops

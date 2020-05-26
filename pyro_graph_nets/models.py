from functools import partial
from pyro_graph_nets.utils.graph_tuple import replace_key, GraphTuple
import torch
from pyro_graph_nets.blocks import NodeBlock, EdgeBlock, GlobalBlock
from torch import nn


def gt_wrap_replace(func):
    def forward(gt):
        v, e, u = func(gt.node_attr, gt.edges.T, gt.edge_attr, gt.global_attr, gt.node_indices, gt.edge_indices)
        gt = replace_key(gt, {
            'node_attr': v,
            'edge_attr': e,
            'global_attr': u
        })
        return gt

    return forward


class GraphAbstractModel(nn.Module):

    def __init__(self, edge_model: nn.Module, node_model: nn.Module,
                 global_model: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleDict({
            'edge_model': edge_model,
            'node_model': node_model,
            'global_model': global_model
        })
        self.reset_parameters()
        self.forward = gt_wrap_replace(self.forward_helper)


    def reset_parameters(self):

        for mod in self.blocks.values():
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()




class GraphEncoder(GraphAbstractModel):

    def forward_helper(self, node_attr, connectivity, edge_attr=None, u=None, node_idx=None, edge_idx=None):

        edge_model = self.blocks['edge_model']
        node_model = self.blocks['node_model']
        global_model = self.blocks['global_model']

        row, col = connectivity

        if edge_model is not None:
            edge_attr = edge_model(node_attr[row], node_attr[col], edge_attr, u, node_idx, edge_idx)
        else:
            edge_attr = None

        if node_model is not None:
            node_attr = node_model(node_attr, connectivity, edge_attr, u, node_idx, edge_idx)
        else:
            node_attr = None

        if global_model is not None:
            global_attr = global_model(node_attr, connectivity, edge_attr, u, node_idx, edge_idx)
        else:
            global_attr = torch.zeros_like(u)

        return node_attr, edge_attr, global_attr


class GraphNetwork(GraphAbstractModel):


    def forward_helper(self, node_attr, edge_index, edge_attr=None, u=None, node_idx=None, edge_idx=None):
        row, col = edge_index

        edge_model = self.blocks['edge_model']
        node_model = self.blocks['node_model']
        global_model = self.blocks['global_model']

        if edge_model is not None:
            try:
                edge_attr = edge_model(node_attr[row], node_attr[col], edge_attr, u, node_idx, edge_idx)
            except RuntimeError as e:
                raise e
                raise type(e)('Edge model error: ' + str(e)) from e
        else:
            edge_attr = torch.zeros_like(edge_attr)

        if node_model is not None:
            try:
                node_attr = node_model(node_attr, edge_index, edge_attr, u, node_idx, edge_idx)
            except RuntimeError as e:
                raise e
                raise type(e)('Node model error: ' + str(e)) from e
        else:
            node_attr = torch.zeros_like(node_attr)

        if global_model is not None:
            global_attr = global_model(node_attr, edge_index, edge_attr, u, node_idx, edge_idx)
        else:
            global_attr = torch.zeros_like(u)

        return node_attr, edge_attr, global_attr



def cat_gt(*gts):
    cat = partial(torch.cat, dim=1)
    return GraphTuple(
        cat([gt.node_attr for gt in gts]),
        cat([gt.edge_attr for gt in gts]),
        cat([gt.global_attr for gt in gts]),
        gts[0].edges,
        gts[0].node_indices,
        gts[0].edge_indices
    )


class OutputTransform(torch.nn.Module):

    def __init__(self, node_fn, edge_fn, global_fn):
        super().__init__()
        self.node_fn = node_fn
        self.edge_fn = edge_fn
        if global_fn is None:
            global_fn = lambda x: x
        self.global_fn = global_fn

    def forward(self, v, connectivity, e, u):
        return self.node_fn(v), self.edge_fn(e), self.global_fn(u)


# TODO: why are the concatenations all mixed up???
class EncoderProcessDecoder(torch.nn.Module):
    """
    enc_vertex_shape=(1, 16),
    core_vertex_shape=(32, 16),
    dec_vertex_shape=(16, 16),
    out_vertex_size=2,

    enc_edge_shape=(1, 16),
    core_edge_shape=(32, 16),
    dec_edge_shape=(16, 16),
    out_edge_size=2,
    device=device,
    """

    def __init__(self, v_features: int,
                 e_features: int,
                 u_features: int,
                 v_out: int,
                 e_out: int,
                 u_out: int):
        super().__init__()

        # in_e_features = n_features * 2 + e_features
        # incoming node features = in_e_features

        latent_size = 16

        # edge features are concatenated along with the src and dest node features
        e_enc_in = e_features + v_features * 2

        # can be anything
        e_enc_out = e_enc_in

        # aggregate the edge_features + node features
        v_enc_in = e_enc_out + u_features + v_features
        v_enc_out = v_enc_in

        # in the core we concatenate the first latent features for some reason??
        e_core_in = e_enc_out * 2
        e_core_out = e_enc_out
        v_core_in = e_core_out + (u_features + v_features) * 2
        v_core_out = v_enc_out

        e_dec_in = 160
        e_dec_out = 160
        v_dec_in = 160
        v_dec_out = 160

        #         e_encoded_in = e_features + v_features * 2
        #         e_encoded_out = e_encoded_in
        #         v_encoded_size =

        self.encoder = GraphEncoder(
            EdgeBlock(e_enc_in, [latent_size, e_enc_out], independent=True),
            NodeBlock(v_enc_in, [latent_size, v_enc_out], independent=True),
            None
        )

        self.core = GraphNetwork(
            EdgeBlock(e_enc_in * 2, [latent_size, e_enc_out], independent=False),
            NodeBlock(v_enc_in * 2, [latent_size, e_enc_out], independent=False),
            None
        )

        self.decoder = GraphEncoder(
            EdgeBlock(11, [16, 11], independent=True),
            NodeBlock(32, [16, 11], independent=True),
            None
        )

        self.output_transform = OutputTransform(
            torch.nn.Linear(5, 5),
            torch.nn.Linear(1, 1),
            None
        )

    @staticmethod
    def graph_tuple_to_args(graph_tuple):
        return (
            graph_tuple.node_attr,
            graph_tuple.edges.T,
            graph_tuple.edge_attr,
            graph_tuple.global_attr
        )

    def forward(self, graph_tuple, steps: int):
        # TODO: copy and detatch?
        encoder = gt_wrap_replace(self.encoder)
        core = gt_wrap_replace(self.core)
        decoder = gt_wrap_replace(self.decoder)
        output_transform = gt_wrap_replace(self.output_transform)
        latent = encoder(graph_tuple)
        latent0 = latent
        output_ops = []
        for _ in range(steps):
            core_input = cat_gt(*[latent0, latent])
            latent = core(core_input)
            decoded_op = output_transform(decoder(latent))
            output_ops.append(decoded_op)
        return output_ops

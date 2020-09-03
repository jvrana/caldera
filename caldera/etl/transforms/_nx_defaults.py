import functools

from ._nx_feature_transform import NetworkxTransformFeatureData


class NetworkxSetDefaultFeature(NetworkxTransformFeatureData):
    def __init__(self, node_default=None, edge_default=None, global_default=None):
        self.node_default = node_default
        self.edge_default = edge_default
        self.global_default = global_default

        setdefault = functools.partial(join_fn=lambda a, b: a, mode="right")

        if node_default is not None:
            node_transform = functools.partial(setdefault, b=node_default)
        else:
            node_transform = None

        if edge_default is not None:
            edge_transform = functools.partial(setdefault, b=edge_default)
        else:
            edge_transform = None

        if global_default is not None:
            global_transform = functools.partial(setdefault, b=global_default)
        else:
            global_transform = None

        super().__init__(
            node_transform=node_transform,
            edge_transform=edge_transform,
            global_transform=global_transform,
        )

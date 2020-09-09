from ._nx_apply_to_key import NetworkxTransformFeatureData
from caldera.utils import functional as fn


def filter_dict(keys):
    if keys is None:
        return None
    return fn.compose(lambda x: x.items(), fn.filter_each(lambda k: k[0] in keys), dict)


class NetworkxFilterDataKeys(NetworkxTransformFeatureData):
    def __init__(self, node_keys=None, edge_keys=None, global_keys=None):
        self.node_keys = node_keys
        self.edge_keys = edge_keys
        self.global_keys = global_keys

        node_transform = filter_dict(node_keys)
        edge_transform = filter_dict(edge_keys)
        glob_transform = filter_dict(global_keys)

        super().__init__(
            node_transform=node_transform,
            edge_transform=edge_transform,
            global_transform=glob_transform,
        )

from copy import deepcopy

from ._nx_feature_transform import NetworkxTransformFeatureData


class NetworkxDeepcopy(NetworkxTransformFeatureData):
    def __init__(self):
        """Deepcopy the graph including all of its node, edge, and global
        features.

        Typically performed before applying other transforms.
        """
        super().__init__(
            node_transform=deepcopy, edge_transform=deepcopy, global_transform=deepcopy
        )

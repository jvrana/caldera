from copy import deepcopy

from ._nx_feature_transform import NetworkxTransformFeatureData


class NetworkxDeepCopyFeatures(NetworkxTransformFeatureData):
    def __init__(self):
        super().__init__(
            node_transform=deepcopy, edge_transform=deepcopy, global_transform=deepcopy
        )

from ._nx_apply_to_key import NetworkxApplyToFeature


class NetworkxFlattenNodeFeature(NetworkxApplyToFeature):
    def __init__(self, key: str):
        super().__init__(key, node_func=lambda x: x.flatten())


class NetworkxFlattenEdgeFeature(NetworkxApplyToFeature):
    def __init__(self, key: str):
        super().__init__(key, edge_func=lambda x: x.flatten())


class NetworkxFlattenGlobalFeature(NetworkxApplyToFeature):
    def __init__(self, key: str):
        super().__init__(key, glob_func=lambda x: x.flatten())

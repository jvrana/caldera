r"""
Methods for transforming :class:`networkx.Graph`s

.. currentmodule:: caldera.transforms.networkx

.. warning::

    You should assume all transforms are performed in place. Use
    :class:`NetworkxDeepcopy` if a copy of the graph is wanted.

.. autosummary::
    :toctree: generated/

    NetworkxApplyToFeature
    NetworkxAttachNumpyFeatures
    NetworkxAttachNumpyOneHot
    NetworkxAttachNumpyBool
    NetworkxSetDefaultFeature
    NetworkxTransformFeatureData
    NetworkxTransformFeatures
    NetworkxFlattenEdgeFeature
    NetworkxFlattenGlobalFeature
    NetworkxFlattenNodeFeature
    NetworkxNodesToStr
    NetworkxToDirected
    NetworkxToUndirected
    NetworkxFilterDataKeys
    NetworkxDeepcpy
"""
from ._nx_apply_to_graph import NetworkxApply
from ._nx_apply_to_key import NetworkxApplyToKey
from ._nx_attach_np_features import NetworkxAttachNumpyBool
from ._nx_attach_np_features import NetworkxAttachNumpyFeatures
from ._nx_attach_np_features import NetworkxAttachNumpyOneHot
from ._nx_deepcopy import NetworkxDeepcopy
from ._nx_defaults import NetworkxSetDefaultFeature
from ._nx_feature_transform import NetworkxTransformFeatureData
from ._nx_feature_transform import NetworkxTransformFeatures
from ._nx_filter_keys import NetworkxFilterDataKeys
from ._nx_flatten_feature import NetworkxFlattenEdgeFeature
from ._nx_flatten_feature import NetworkxFlattenGlobalFeature
from ._nx_flatten_feature import NetworkxFlattenNodeFeature
from ._nx_nodes_to_str import NetworkxNodesToStr
from ._nx_to_directed import NetworkxToDirected
from ._nx_to_undirected import NetworkxToUndirected


__all__ = [
    "NetworkxAttachNumpyFeatures",
    "NetworkxAttachNumpyOneHot",
    "NetworkxAttachNumpyBool",
    "NetworkxSetDefaultFeature",
    "NetworkxTransformFeatureData",
    "NetworkxTransformFeatures",
    "NetworkxNodesToStr",
    "NetworkxToDirected",
    "NetworkxToUndirected",
    "NetworkxApplyToKey",
    "NetworkxFlattenNodeFeature",
    "NetworkxFlattenEdgeFeature",
    "NetworkxFlattenGlobalFeature",
    "NetworkxFilterDataKeys",
    "NetworkxApply",
    "NetworkxDeepcopy",
]

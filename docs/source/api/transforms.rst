Transforms
==========

.. currentmodule:: caldera.transforms

Methods for transforming GraphData and GraphBatch

GraphData and GraphBatch
------------------------

Transformation classes for :class:`caldera.data.GraphData` and :class:`caldera.data.GraphBatch`.

.. autosummary::
    :toctree: ../generated/

   Shuffle
   Reverse
   Undirected
   FullyConnected
   RandomEdgeMask
   RandomNodeMask
   RandomHop

Preprocessing Transforms
------------------------

Networkx
^^^^^^^^

Methods for transforming :class:`networkx.Graph`

.. autosummary::
    :toctree: ../generated/

    networkx.NetworkxApplyToFeature
    networkx.NetworkxAttachNumpyFeatures
    networkx.NetworkxAttachNumpyOneHot
    networkx.NetworkxSetDefaultFeature
    networkx.NetworkxTransformFeatureData
    networkx.NetworkxTransformFeatures
    networkx.NetworkxFlattenEdgeFeature
    networkx.NetworkxFlattenGlobalFeature
    networkx.NetworkxFlattenNodeFeature
    networkx.NetworkxNodesToStr
    networkx.NetworkxToDirected
    networkx.NetworkxToUndirected
    networkx.NetworkxFilterDataKeys
    networkx.NetworkxDeepCopyFeatures

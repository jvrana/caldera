# Pyro-GraphNets

Pytorch/Pyro implementation of DeepMind's graphnets library.

Original TensorFlow1 non-Bayesian implementation can be found at [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261) and at [DeepMind's github repository](https://github.com/deepmind/graph_nets).

Goals of this library are to:

* provide a simple API for training deep neural networks on graph data
* implement bayesian neural networks on graph data

## Installation

You'll required `conda` installed. Load the conda environment by running:

```bash
make
```

Check installation:

```bash
make check
```

Lock the environment:

```bash
make lock
```

### installing `torch_scatter`

## Tour

##### GraphTuple

The `GraphTuple` an object borrowed from deepminds/graph_nets 
that represents MultiGraphs and MultiDiGraphs. The `GraphTuple` is a tuple of
`tensor.tensor` objects represnting multigraphs (directed and undirected).

Dimensions:

`gt.node_attr` - [n_nodes, num_node_features]

`gt.edge_attr` - [n_edges, num_edge_features]

`gt.global_attr` - [n_graphs, num_global_features]

`gt.edges` - [n_edges, 2]

`gt.node_indices` - [n_nodes]

`gt.edge_indices` - [n_edges]

To convert a list of netorkx graphs into GraphTuple

```python
from archived.pyro_graph_nets.utils import to_graph_tuple

# expect 'features' key on node_attributes
# expect 'features' key on edge_attributes
# expect 'features' key on `graph.data` attribute

gt = to_graph_tuple(graphs, feature_key='features')
```

For input and target data, it is easy to keep different features on the same graph 
and pull out the relevant data using a specified key:

```python
input_gt = to_graph_tuple(graphs, feature_key='features')
target_gt = to_graph_tuple(graphs, feature_key='target')
```

##### FlexAPI

The `Flex` API allows automatic setting of input dimensions.
A simple example demonstrates this.

```python
from archived.pyro_graph_nets import Flex
from torch import nn
from torch import tensor

flex_linear = Flex(nn.Linear)(Flex.d(), 11)

# input size for Linear is unknown until given data
print(flex_linear)

"""
FlexBlock()
"""

x = tensor.randn((10, 55))

flex_linear(x)

"""
FlexBlock(
  (resolved_module): Linear(in_features=55, out_features=11, bias=True)
)
"""
```

##### GraphDataLoader

**Random graph generation**

```python
# TODO: examples of generating random graphs for demo and testing purposes.
```

`GraphDataLoader` and `GraphDataset` provides graph data for 

```python
from archived.pyro_graph_nets.utils import GraphDataLoader, GraphDataset
from archived.pyro_graph_nets.utils import to_graph_tuple

dataset = GraphDataset(graphs)
dataloader = GraphDataLoader(dataset, batch_size=10, shuffle=True)
for batch_ndx, sample for enumerate(dataloader):
    input_gt = to_graph_tuple(sample, feature_key='features')
    target_gt = to_graph_tuple(sample, feature_key='target')
```

### Graph Neural Network implementations

[Battaglia et. al. 2018](https://arxiv.org/pdf/1806.01261.pdf)

* Full GN
* Message Passing
* Non-local Neural Network
* Deep set
* Relation networks
* Independent recurrent neural network

## Example

TODO: show example where any layers can be added to a graph network.

Custom types of aggregation layers

Custom blocks

### Hyperparameter Optimization

* optimization on neural network shape
* optimization of aggregation methods and options

### Non-Bayesian Examples

### Bayesian Graph Neural Nets

#### Regulatory Network Predictions from uncertain scRNA-Seq Data

**Goal**: Given examples of regulatory networks (as graphs) and scRNA-seq data, predict the scRNA-seq data for a *new* regulatory network.

#### Regulatory Network Kinetics

Given a steady state prediction, use some kinetics prior to simulate kinetics of new regulatory networks.

#### Synthetic Regulatory Network Predictions

**Goal**: Given example of synthetic regulatory networks (as graphs) and scRNA-seq data

## Acknowledgements

* [rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) - an awesome library that provides missing 
scatter methods in pytorch. Necessary for graph attribute aggregation
* PyTorch Geometric - 
* PGN - provided a blueprint for implementing this library

## References



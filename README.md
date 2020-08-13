# Caldera 🌋

**SOFTWARE IS IN ALPHA, README IS PENDING**

**Caldera** is a Pytorch extension for learning on graphical data. It learns underlying causal relationships in data.

**Inspirations**

* Pytorch Geometric [ADD REF]
* GraphNets [ADD REF]
* Pyro
* Judea Pearl [ADD REF]

## Gallery

1. TODO: chemical reaction network oscillator design
2. TODO: linkage prediction on CORA dataset
3. TODO: 

## Usage

### Installation

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

### Tour

#### GraphData, GraphBatch, GraphDataLoader

The `GraphData` an object borrowed heavily from deepminds/graph_nets and 
Pytorch Geometric that represents MultiGraphs and MultiDiGraphs. 
The `GraphData` is a tuple of `tensor.tensor` objects represnting multigraphs 
(directed and undirected).

Dimensions:

`gt.x` - [n_nodes, num_node_features]

`gt.e` - [n_edges, num_edge_features]

`gt.g` - [n_graphs, num_global_features]

`gt.edges` - [n_edges, 2]

To support minbatch training, the `GraphBatch` represents a batch of
GraphData instances and has the following additional tensors:

`gt.node_idx` - [n_nodes]

`gt.edge_idx` - [n_edges]

Conversion between GraphData lists and GraphBatch instances are supported
as well as conversion between GraphData and GraphBatch to and from networkx objects.

The `GraphDataLoader` class provides a data loader for returning `GraphBatch`
instances from a list of `GraphData` objects.

```python
from caldera.data import GraphDataLoader

loader = GraphDataLoader(data, batch_size=32, shuffle=True)

for graph_batch in loader:
    pass # training loop here
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

### Simulating Chemical Reaction Networks


Target: randomized non-cyclic circuits and steady-state


* learning circuit behavior from steady-state data
*  
* ... with unexpected cross-talk of parts
* ... with part specific toxicity simulated
* simulating uncertainty

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



# Pyro-GraphNets

Pytorch/Pyro implementation of DeepMind's graphnets library.

Original TensorFlow1 non-Bayesian implementation can be found at [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261) and at [DeepMind's github repository](https://github.com/deepmind/graph_nets).

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

##### FlexAPI

The `Flex` API allows automatic setting of input dimensions.
A simple example demonstrates this.

```python
from pyro_graph_nets.flex import Flex
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



```python
from pyro_graph_nets.models import GraphNetwork
from pyro_graph_nets.blocks import FlexBlock, EdgeBlock, MLP
model = GraphNetwork(
    FlexBlock(EdgeBlock, MLP(FlexBlock.dim(), 16, 20))
)
```

##### GraphDataLoader

**Random graph generation**

```python
# TODO: examples of generating random graphs for demo and testing purposes.
```

`GraphDataLoader` and `GraphDataset` provides graph data for 

```python
from pyro_graph_nets.utils.data import GraphDataLoader, GraphDataset
from pyro_graph_nets.utils.graph_tuple import to_graph_tuple

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

### Non-Bayesian Examples

### Bayesian Graph Neural Nets

#### Regulatory Network Predictions from uncertain scRNA-Seq Data

**Goal**: Given examples of regulatory networks (as graphs) and scRNA-seq data, predict the scRNA-seq data for a *new* regulatory network.

#### Regulatory Network Kinetics

Given a steady state prediction, use some kinetics prior to simulate kinetics of new regulatory networks.

#### Synthetic Regulatory Network Predictions

**Goal**: Given example of synthetic regulatory networks (as graphs) and scRNA-seq data

## Acknowledgements

* [rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)
* PyTorch Geometric
* pgn

## References



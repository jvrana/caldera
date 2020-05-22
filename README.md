# Pyro-GraphNets

Pytorch/Pyro implementation of DeepMind's graphnets library.

Original TensorFlow1 non-Bayesian implementation can be found at [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261) and at [DeepMind's github repository](https://github.com/deepmind/graph_nets).

## Example

### Non-Bayesian Examples

### Bayesian Graph Neural Nets

#### Regulatory Network Predictions from uncertain scRNA-Seq Data

**Goal**: Given examples of regulatory networks (as graphs) and scRNA-seq data, predict the scRNA-seq data for a *new* regulatory network.

#### Regulatory Network Kinetics

Given a steady state prediction, use some kinetics prior to simulate kinetics of new regulatory networks.

#### Synthetic Regulatory Network Predictions

**Goal**: Given example of synthetic regulatory networks (as graphs) and scRNA-seq data


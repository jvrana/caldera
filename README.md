# Caldera 🌋

![Code Quality](https://github.com/jvrana/caldera/workflows/Code%20Quality/badge.svg)
![Tests](https://github.com/jvrana/caldera/workflows/Tests/badge.svg)
![Documentation](https://github.com/jvrana/caldera/workflows/Documentation/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**SOFTWARE IS IN ALPHA, README IS PENDING**

**Caldera** is a Pytorch extension for learning on graphical data. It learns underlying causal relationships in data.

**Inspirations**

* Pytorch Geometric [ADD REF]
* GraphNets [ADD REF]
* Pyro Probabilistic Programming Language
* Pearl, J. (2009). Causality: Models, Reasoning and Inference.
* Pearl, J. (2018). Theoretical impediments to machine learning with seven sparks from the causal revolution.

## Gallery

## Usage

### Installation

#### via pip

#### via Conda

#### via Docker

**Build CUDA enable image**

```
docker build . -f docker/cu101/Dockerfile
```

**Build slim cpu-only image**

```
docker build . -f docker/cpu/Dockerfile
```

### Tour

## Acknowledgements

* [rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) - an awesome library that provides missing 
scatter methods in pytorch. Necessary for graph attribute aggregation
* PyTorch Geometric - 
* PGN - provided a blueprint for implementing this library

## References

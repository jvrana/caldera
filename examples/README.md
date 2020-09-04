# Examples

## Supervised Examples

### Traversals

In these examples, we use **caldera** to find the unweighted 
shortest path, 
weighted shortest path,
minimum spanning tree,
minimum flow,
and minimum steiner tree using labeled graphs.

### Function Networks

In these examples, we use **caldera** to approximate a complex network of functions.
We apply this to the problem of chemical reaction networks in order to learn
a mapping from a *genetic part* to *behavior*.

### Link Prediction

In these examples, we use **caldera** in various linkage prediction tasks. 
We apply it to the biological network data (*E. coli*, *S. cerevisiae*) and
other networks (e.g. CORA).

## Unsupervised Examples

### Algorithm approximation (a.k.a. deep program synthesis)

In this example, we use **caldera** to approximate arbitrary 
algorithms, using the concept of *graphs* to represent 
control flow statements in an algorithm. This example showcases
generative graph networks, graph autoencoders, and graph attention networks.

We apply this to approximate a solution for **NP-complete** problems by providing a *verification* function
to the output of our networks and using that as a cost function in our unsupervised training.
We allow the network to determine its internal graph structure (data control flow).

### Encoders

...

### Graph Clustering
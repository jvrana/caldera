"""summary.py.

Methods for summaries and statistics
"""
import numpy as np
import torch

from caldera.data import GraphDataLoader


def degree(graph_batch, i):
    number_of_nodes = graph_batch.x.shape[0]
    counts = torch.unique(graph_batch.edges[i], return_counts=True)[1]
    degree = torch.cat(
        [counts, torch.zeros(number_of_nodes - counts.shape[0], dtype=torch.long)]
    ).to(torch.float)
    return degree.mean(), degree.std()


def loader_summary(loader: GraphDataLoader):
    x = []
    e = []
    g = []
    indegrees = []
    outdegrees = []
    for d, _ in loader:
        number_of_nodes = d.x.shape[0]
        number_of_edges = d.e.shape[0]
        number_of_graphs = d.g.shape[0]

        x.append(number_of_nodes)
        e.append(number_of_edges)
        g.append(number_of_graphs)

        indegrees.append(degree(d, 1)[0])
        outdegrees.append(degree(d, 0)[0])

    x = np.array(x)
    e = np.array(e)
    g = np.array(g)
    indegrees = np.array(indegrees)
    outdegrees = np.array(outdegrees)

    info = {
        "graphs": {
            "sum": g.sum(),
            "batch_mean": g.mean(),
            "batch_std": g.std(),
            "batch_indegree_mean": indegrees.mean(),
            "batch_outdegree_mean": outdegrees.mean(),
            "batch_indegree_std": indegrees.std(),
            "batch_outdegree_std": outdegrees.std(),
        },
        "edges": {"sum": e.sum(), "batch_mean": e.mean(), "batch_std": e.std()},
        "nodes": {"sum": x.sum(), "batch_mean": x.mean(), "batch_std": x.std()},
    }

    return info

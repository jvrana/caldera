import networkx as nx
import pandas as pd
import seaborn as sns
import torch
from pylab import plt

from .graph_plotter import GraphPlotter


def plot_example_graphs(graphs, rows=4, cols=4, figsize=(10, 10)):
    import matplotlib as mplt

    norm = mplt.colors.Normalize(0, 20)

    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=figsize)

    def flatten(x):
        arr = []
        for a in x:
            for b in a:
                arr.append(b)
        return arr

    axes = flatten(axes)
    for g, ax in zip(graphs, axes):
        if not g.number_of_nodes():
            continue
        ylabels = {n: ndata for n, ndata in g.nodes(data="y")}
        nodelist, nodedata = zip(*g.nodes(data=True))
        ylabels = {
            n: round(ndata["y"].flatten().item(), 2)
            for n, ndata in zip(nodelist, nodedata)
        }
        y = [round(ndata["y"].flatten().item(), 2) for ndata in nodedata]

        pos = GraphPlotter._topological_sort(g)
        nx.draw_networkx_labels(g, pos, labels=ylabels, ax=ax)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=mplt.cm.coolwarm(norm(y)))
        nx.draw_networkx_edges(g, pos, ax=ax)
    return fig, axes


def plot_kinetics(model, data, steps):
    with torch.no_grad():
        out = model(data, steps)
        x_arr = []
        for i, data in enumerate(out):
            t = torch.tensor([[i]] * data.x.shape[0], dtype=torch.float)
            index = torch.unsqueeze(torch.arange(data.x.shape[0], dtype=torch.float), 1)
            index = index.to(data.x.device)
            t = t.to(data.x.device)
            x = torch.cat([t, data.x, index], dim=1)
            x_arr.append(x)

        x = torch.cat(x_arr)
        df = pd.DataFrame(
            {
                "t": x[:, 0].detach().cpu(),
                "x": x[:, 1].detach().cpu(),
                "node": x[:, 2].detach().cpu(),
            }
        )
        df = df[df["node"] < 20]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax.set_ylim(-5, 20)
        sns.lineplot(x="t", y="x", hue="node", data=df, ax=ax)

    return ax, fig

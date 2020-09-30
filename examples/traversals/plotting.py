import io
from contextlib import contextmanager

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from PIL import Image


def edge_colors(g, key, cmap):
    edgelist = list(g.edges)
    edgefeat = list()
    for e in edgelist:
        edata = g.edges[e]
        edgefeat.append(edata[key][0].item())
    edgefeat = np.array(edgefeat)
    edgecolors = cmap(edgefeat)
    return edgecolors
    nx.draw_networkx_edges(g, pos=pos, edge_color=edgecolors, arrows=False)


def node_colors(g, key, cmap):
    nodelist = list(g.nodes)
    nodefeat = list()
    for n in nodelist:
        ndata = g.nodes[n]
        nodefeat.append(ndata[key][0].item())
    nodefeat = np.array(nodefeat)
    nodecolors = cmap(nodefeat)
    return nodecolors
    nx.draw_networkx_nodes(g, pos=pos, node_size=10, node_color=nodecolors)


def plot_graph(g, ax, cmap, key="features", seed=1):
    pos = nx.layout.spring_layout(g, seed=seed)
    nx.draw_networkx_edges(
        g, ax=ax, pos=pos, edge_color=edge_colors(g, key, cmap), arrows=False, width=1.5
    )
    nx.draw_networkx_nodes(
        g, ax=ax, pos=pos, node_size=5, node_color=node_colors(g, key, cmap)
    )


def comparison_plot(out_g, expected_g):
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    axes[0].axis("off")
    axes[1].axis("off")

    axes[0].set_title("out")
    plot_graph(out_g, axes[0], cm.plasma)

    axes[1].set_title("expected")
    plot_graph(expected_g, axes[1], cm.plasma)
    return fig, axes


def validate_compare_plot(trainer, plmodel):
    eval_loader = trainer.val_dataloaders[0]
    for x, y in eval_loader:
        break
    plmodel.eval()
    y_hat = plmodel.model.forward(x, 10)[-1]
    y_graphs = y.to_networkx_list()
    y_hat_graphs = y_hat.to_networkx_list()

    idx = 0
    yg = y_graphs[idx]
    yhg = y_hat_graphs[idx]
    return comparison_plot(yhg, yg)


@contextmanager
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)
    yield im
    buf.close()
    fig.close()


@contextmanager
def figs_to_pils(figs):
    ims = []
    buffs = []
    for fig in figs:
        buf = io.BytesIO()
        buffs.append(buf)
        fig.savefig(buf, format="png")
        buf.seek(0)
        ims.append(Image.open(buf))
    yield ims
    for buf in buffs:
        buf.close()
    plt.close()

from caldera.defaults import CalderaDefaults
from caldera.utils.nx.types import Graph


def _flatten_values(datalist):
    for data in datalist:
        for k, v in data.items():
            if hasattr(v, "flatten"):
                v = v.flatten()
            data[k] = v


def flatten_nodes(g: Graph):
    _flatten_values([x[-1] for x in g.nodes(data=True)])


def flatten_edges(g: Graph):
    _flatten_values([x[-1] for x in g.edges(data=True)])


def flatten_global(g: Graph, global_key: str = None):
    _flatten_values([g.get_global(global_key)])

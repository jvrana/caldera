from typing import Dict
from typing import Optional
from typing import TypeVar
from typing import Union

from caldera.defaults import CalderaDefaults
from caldera.utils import dict_join
from caldera.utils.nx.types import Graph


T = TypeVar("T")
S = TypeVar("S")
K = TypeVar("K")


def setdefault_inplace(d1: Dict[K, T], d2: Dict[K, S]) -> Dict[K, Union[T, S]]:
    return dict_join(d1, d2, d1, join_fn=lambda a, b: a, mode="right")


def add_default_node_data(g: Graph, data: Dict):
    """Update set default node data.

    Will not update if key exists in data.
    """
    for _, ndata in g.nodes(data=True):
        setdefault_inplace(ndata, data)


def add_default_edge_data(g: Graph, data: Dict):
    """Update set default edge data.

    Will not update if key exists in data.
    """
    for _, _, edata in g.edges(data=True):
        setdefault_inplace(edata, data)


def add_default_global_data(
    g: Graph, data: Dict, global_key: str = CalderaDefaults.nx_global_key
):
    """Update set default glboal data.

    Will not update if key exists in data.
    """
    if not g.get_global(global_key):
        g.set_global(data, global_key)


def add_default(
    g: Graph,
    *,
    node_data: Optional[Dict] = None,
    edge_data: Optional[Dict] = None,
    global_data: Optional[Dict] = None,
    global_key: str = None
):
    if node_data:
        add_default_node_data(g, node_data)
    if edge_data:
        add_default_edge_data(g, edge_data)
    if global_data:
        add_default_global_data(g, global_data, global_key)

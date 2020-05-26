
from torch.utils.data import DataLoader, Dataset
from pyro_graph_nets.utils.graph_tuple import to_graph_tuple, GraphTuple
import networkx as nx
from typing import Union
from typing import List
from typing import Generator, Callable, Any, Tuple


GraphType = Union[nx.DiGraph, nx.Graph]


def random_graph_generator(
        n_nodes: Union[int, Callable[[], int]],
        theta: int,
        n_feat_gen: Callable[[], Any],
        e_feat_gen: Callable[[], Any],
        g_feat_gen: Callable[[], Any],
        attribute_name: str = 'features'
) -> Generator[GraphType, None, None]:
    def _resolve(x):
        if callable(x):
            return x()
        return x

    while True:
        n_nodes = _resolve(n_nodes)
        _theta = _resolve(theta)
        g = nx.geographical_threshold_graph(n_nodes, _theta)
        dg = nx.DiGraph()
        dg.add_edges_from(g.edges)
        add_features(dg,
                     n_feat_gen,
                     e_feat_gen,
                     g_feat_gen,
                     attribute_name=attribute_name)
        yield dg


def add_features(g: GraphType,
                 n_feat_gen: Callable[[], Any],
                 e_feat_gen: Callable[[], Any],
                 g_feat_gen: Callable[[], Any],
                 attribute_name: str = 'features'):
    """
    Add features to a graph
    :param g:
    :param n_feat_gen:
    :param e_feat_gen:
    :param g_feat_gen:
    :param attribute_name:
    :return:
    """
    for _, ndata in g.nodes(data=True):
        ndata[attribute_name] = n_feat_gen()
    for _, _, edata in g.edges(data=True):
        edata[attribute_name] = e_feat_gen()

    if not hasattr(g, 'data'):
        g.data = {}
    g.data.update({attribute_name: g_feat_gen()})

    return g


def random_input_output_graphs(
        n_nodes: Union[Callable[[], int], int],
        theta: int,
        n_feat_gen0: Callable[[], Any],
        e_feat_gen0: Callable[[], Any],
        g_feat_gen0: Callable[[], Any],
        n_feat_gen1: Callable[[], Any],
        e_feat_gen1: Callable[[], Any],
        g_feat_gen1: Callable[[], Any],
        input_attr_name: str = 'features',
        target_attr_name: str = 'features',
        do_copy: bool = True) \
        -> Generator[Tuple[GraphType, GraphType], None, None]:
    """
    Create a generator of randomized input and target graphs. Randomly attach node, edge,
    or global features using the provided generators. Global attributes can be found
    on using `graph.data['features']`.

    :param n_nodes: number of nodes or a generator to return the number of nodes
    :param theta: threshold branching factor (lower means more connected, 1000 is trees)
    :param n_feat_gen0: callable for input node features (e.g. `np.random.randint(0, 10, 5)`)
    :param e_feat_gen0: callable for input node features (e.g. `np.random.randint(0, 10, 5)`)
    :param g_feat_gen0: callable for input node features (e.g. `np.random.randint(0, 10, 5)`)
    :param n_feat_gen1: callable for input node features (e.g. `np.random.randint(0, 10, 5)`)
    :param e_feat_gen1: callable for input node features (e.g. `np.random.randint(0, 10, 5)`)
    :param g_feat_gen1: callable for input node features (e.g. `np.random.randint(0, 10, 5)`)
    :param input_attr_name: default attribute name for the node, edge, and graph data (default: 'features')
    :param target_attr_name: default attribute name for the node, edge, and graph data (default: 'target')
    :return: generator of the two graphs.
    """

    if do_copy is False and input_attr_name == target_attr_name:
        raise ValueError("Input and target attribute names cannot be equal when do_copy=False.")

    def make_target(g: GraphType) -> GraphType:
        if do_copy:
            g_copy = type(g)()
            g_copy.add_nodes_from(g.nodes)
            g_copy.add_edges_from(g.edges)
        else:
            g_copy = g
        add_features(
            g_copy,
            n_feat_gen1,
            e_feat_gen1,
            g_feat_gen1,
            attribute_name=target_attr_name
        )
        return g_copy

    gen = random_graph_generator(n_nodes,
                                 theta,
                                 n_feat_gen0,
                                 e_feat_gen0,
                                 g_feat_gen0,
                                 attribute_name=input_attr_name)
    for input_g in gen:
        target_g = make_target(input_g)
        if do_copy:
            yield input_g, target_g
        else:
            yield target_g


class GraphDataset(Dataset):
    """Dataset to hold graphs"""

    def __init__(self, graphs: List[GraphType]):
        """
        Construct a graph dataset

        :param graphs: list of networkx graphs
        """
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class GraphDataLoader(DataLoader):

    def __init__(self, *args,
                 **kwargs):
        _kwargs = {'collate_fn': self.collate_graphs}
        _kwargs.update(kwargs)
        super().__init__(*args, **_kwargs)

    @staticmethod
    def collate_graphs(graphs: List[GraphType], ) -> GraphTuple:
        return list(graphs)


class GraphTupleDataLoader(DataLoader):

    def __init__(self, *args,
                 feature_key: str = 'features',
                 global_attr_key: str = 'data',
                 **kwargs):
        """
        DataLoader for graphs. Automatic batching results in batching of
        graphs into single GraphTuple instances.

        Example usage:

        .. code-block:: python

            dataset = GraphDataset(graphs)
            dataloader = GraphDataLoader(dataset, shuffle=True, batch_size=10)

        :param args: dataloader args
        :param kwargs: dataloader kwargs
        """
        self._feature_key = feature_key
        self._global_attr_key = global_attr_key
        _kwargs = {'collate_fn': self.collate_graphs_to_graph_tuple}
        _kwargs.update(kwargs)
        super().__init__(*args, **_kwargs)

    def collate_graphs_to_graph_tuple(self, graphs: List[GraphType], ) -> GraphTuple:
        return to_graph_tuple(graphs, feature_key=self._feature_key,
                              global_attr_key=self._global_attr_key)
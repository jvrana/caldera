from pyrographnets.data import GraphData, GraphBatch
import torch
import pytest
import networkx as nx
from pyrographnets.data.utils import random_data
from flaky import flaky


class Comparator:

    @staticmethod
    def data_to_nx(data, g, fkey, gkey):
        """Compare `GraphData` to `nx.DiGraph` instance."""
        assert data.x.shape[0] == g.number_of_nodes(), 'Check number of nodes'

        assert data.e.shape[0] == g.number_of_edges(), 'Check number of edges'

        assert data.edges.shape[1] == g.number_of_edges(), 'Check edges vs edge attr'

        # ensure feature key is in node data
        for _, ndata in g.nodes(data=True):
            assert fkey in ndata

        # ensure feature key is in edge data
        for _, _, edata in g.edges(data=True):
            assert fkey in edata

        # ensure feature key is in global data
        assert hasattr(g, gkey)
        gdata = getattr(g, gkey)[fkey]
        assert gdata is not None

        # check edges
        src = []
        dest = []
        ndict = {}
        for i, n in enumerate(g.nodes()):
            ndict[n] = i
        for n1, n2, ne in g.ordered_edges:
            src.append(ndict[n1])
            dest.append(ndict[n2])
        edges = torch.tensor([src, dest])
        print(edges)
        print(data.edges)

        assert torch.all(torch.eq(edges, data.edges))

        # check global data
        assert torch.all(torch.eq(gdata, data.g))
        assert gdata is not data.g

        # check node data
        nodes = list(g.nodes(data=True))
        for i in range(len(nodes)):
            assert torch.all(torch.eq(nodes[i][1][fkey], data.x[i]))

        # check edge data
        for i, (n1, n2, ne) in enumerate(g.ordered_edges):
            edata = g[n1][n2][ne]
            assert torch.all(torch.eq(edata[fkey], data.e[i]))


class TestGraphData:

    def test_graph_data_init_0(self):
        data = GraphData(
            torch.randn(10, 5),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3]))
        )
        assert data.x.shape == torch.Size([10, 5])
        assert data.e.shape == torch.Size([3, 4])
        assert data.edges.shape == torch.Size([2, 3])
        assert data.g.shape == torch.Size([1, 3])

    def test_graph_data_init_1(self):
        data = GraphData(
            torch.randn(10, 5),
            torch.randn(5, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 5]))
        )
        assert data.x.shape == torch.Size([10, 5])
        assert data.e.shape == torch.Size([5, 4])
        assert data.edges.shape == torch.Size([2, 5])
        assert data.g.shape == torch.Size([1, 3])

    @pytest.mark.parametrize(
        'keys', [
            (None, None),
            ('myfeatures', 'mydata'),
            ('features', 'data')
        ]
    )
    def test_to_networkx(self, keys):
        kwargs = {
            'feature_key': 'features',
            'global_attr_key': 'data'
        }
        feature_key, global_attr_key = keys
        if feature_key is not None:
            kwargs['feature_key'] = feature_key
        else:
            del kwargs['feature_key']
        if global_attr_key is not None:
            kwargs['global_attr_key'] = global_attr_key
        else:
            del kwargs['global_attr_key']

        data = GraphData(
            torch.randn(10, 5),
            torch.randn(5, 4),
            torch.randn(1, 3),
            torch.tensor([
                [0, 1, 2, 3, 4],
                [4, 3, 2, 1, 0]
            ])
        )

        g = data.to_networkx(**kwargs)
        assert isinstance(g, nx.OrderedMultiDiGraph)
        assert g.number_of_nodes() == 10
        assert g.number_of_edges() == 5

        fkey = kwargs.get('feature_key', 'features')
        gkey = kwargs.get('global_attr_key', 'data')

        Comparator.data_to_nx(data, g, fkey, gkey)

    @pytest.mark.parametrize(
        'keys', [
            (None, None),
            ('myfeatures', 'mydata'),
            ('features', 'data')
        ]
    )
    def test_from_networkx(self, keys):
        kwargs = {
            'feature_key': 'features',
            'global_attr_key': 'data'
        }
        feature_key, global_attr_key = keys
        if feature_key is not None:
            kwargs['feature_key'] = feature_key
        else:
            del kwargs['feature_key']
        if global_attr_key is not None:
            kwargs['global_attr_key'] = global_attr_key
        else:
            del kwargs['global_attr_key']

        fkey = kwargs.get('feature_key', 'features')
        gkey = kwargs.get('global_attr_key', 'data')

        g = nx.OrderedMultiDiGraph()
        g.add_node('node1', **{fkey: torch.randn(5)})
        g.add_node('node2', **{fkey: torch.randn(5)})
        g.add_edge('node1', 'node2', **{fkey: torch.randn(4)})
        g.ordered_edges = [
            ('node1', 'node2', 0)
        ]
        setattr(g, gkey, {fkey: torch.randn(3)})

        data = GraphData.from_networkx(g, **kwargs)

        Comparator.data_to_nx(data, g, fkey, gkey)

    def test_empty_networkx(self):
        """Empty graphs should be OK"""
        g = nx.DiGraph()
        GraphData.from_networkx(g)

    @pytest.mark.parametrize(
        'keys', [
            (None, None),
            ('myfeatures', 'mydata'),
            ('features', 'data')
        ]
    )
    def test_from_networkx_no_edge(self, keys):
        kwargs = {
            'feature_key': 'features',
            'global_attr_key': 'data'
        }
        feature_key, global_attr_key = keys
        if feature_key is not None:
            kwargs['feature_key'] = feature_key
        else:
            del kwargs['feature_key']
        if global_attr_key is not None:
            kwargs['global_attr_key'] = global_attr_key
        else:
            del kwargs['global_attr_key']

        fkey = kwargs.get('feature_key', 'features')
        gkey = kwargs.get('global_attr_key', 'data')

        g = nx.OrderedMultiDiGraph()
        g.add_node('node1', **{fkey: torch.randn(5)})
        g.add_node('node2', **{fkey: torch.randn(5)})
        g.ordered_edges = []
        # g.add_edge('node1', 'node2', **{fkey: torch.randn(4)})

        setattr(g, gkey, {fkey: torch.randn(3)})

        data = GraphData.from_networkx(g, **kwargs)

        Comparator.data_to_nx(data, g, fkey, gkey)

class TestGraphDataModifiers:

    def test_append_nodes(self):
        data = GraphData(
            torch.randn(10, 5),
            torch.randn(5, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 5]))
        )

        assert data.x.shape[0] == 10
        data.append_nodes(torch.randn(2, 5))
        assert data.x.shape[0] == 12

    def test_append_edges(self):
        data = GraphData(
            torch.randn(10, 5),
            torch.randn(5, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 5]))
        )

        e = torch.randn(3, 4)
        edges = torch.randint(0, 10, torch.Size([2, 3]))
        data.append_edges(e, edges)

    def test_invalid_append_edges(self):
        data = GraphData(
            torch.randn(10, 5),
            torch.randn(5, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 5]))
        )

        e = torch.randn(3, 4)
        edges = torch.randint(0, 10, torch.Size([2, 4]))
        with pytest.raises(RuntimeError):
            data.append_edges(e, edges)




class TestGraphBatchModifiers:

    def test_batch_append_nodes(self):

        datalist = [random_data(5, 6, 7) for _ in range(10)]
        batch = GraphBatch.from_data_list(datalist)

        x = torch.randn(3, 5)
        idx = torch.tensor([0, 1, 2])

        node_shape = batch.x.shape
        batch.append_nodes(x, idx)
        print(batch.node_idx.shape)

        assert node_shape[0] < batch.x.shape[0]
        print(batch.node_idx.shape)
        print(batch.x.shape)


    def test_change_connectivity(self):
        """Tests changing the connectivity of the batch"""
        datalist = [random_data(5, 3, 4) for _ in range(2000)]
        batch = GraphBatch.from_data_list(datalist)

        a = torch.randint(0, 2, (batch.e.shape[0],))

        # begin tracking gradients
        a.requires_grad = True

        b = torch.where(a == 1)
        e = batch.e
        edge_idx = batch.edge_idx
        edges = batch.edges

        # mask connectivity
        edges = edges.T[b].T
        edge_idx = edge_idx[b]
        e = e[b]

        assert edges.shape[1] < batch.edges.shape[1]
        assert e.shape[0] < batch.e.shape[0]
        assert edge_idx.shape[0] < batch.edge_idx.shape[0]

        new_batch = GraphBatch(batch.x, e, batch.g, edges, batch.node_idx, edge_idx)

        assert new_batch.edges.shape[1] < batch.edges.shape[1]

    @pytest.mark.parametrize('attr', ['x', 'e', 'g'])
    def test_is_differentiable__to_datalist(self, attr):
        datalist = [random_data(5, 3, 4) for _ in range(300)]

        batch = GraphBatch.from_data_list(datalist)

        getattr(batch, attr).requires_grad = True

        datalist = batch.to_data_list()
        for data in datalist:
            assert getattr(data, attr).requires_grad

    @pytest.mark.parametrize('attr', ['x', 'e', 'g'])
    def test_is_differentiable__from_datalist(self, attr):
        datalist = [random_data(5, 3, 4) for _ in range(300)]
        for data in datalist:
            getattr(data, attr).requires_grad = True
        batch = GraphBatch.from_data_list(datalist)
        assert getattr(batch, attr).requires_grad

    @pytest.mark.parametrize('attr', ['x', 'e', 'g'])
    def test_is_differentiable__append_nodes(self, attr):
        datalist = [random_data(5, 3, 4) for _ in range(300)]
        for data in datalist:
            getattr(data, attr).requires_grad = True
        batch = GraphBatch.from_data_list(datalist)

        new_nodes = torch.randn(10, 5)
        idx = torch.ones(10, dtype=torch.long)
        n_nodes = batch.x.shape[0]
        batch.append_nodes(new_nodes, idx)
        assert batch.x.shape[0] == n_nodes + 10
        assert getattr(batch, attr).requires_grad


    @pytest.mark.parametrize('attr', ['x', 'e', 'g'])
    def test_is_differentiable__append_edges(self, attr):
        datalist = [random_data(5, 3, 4) for _ in range(300)]
        for data in datalist:
            getattr(data, attr).requires_grad = True
        batch = GraphBatch.from_data_list(datalist)

        new_edge_attr = torch.randn(20, 3)
        new_edges = torch.randint(0, batch.x.shape[0], (2, 20))
        idx = torch.randint(0, 30, (new_edges.shape[1],))
        idx = torch.sort(idx).values

        n_edges = batch.e.shape[0]
        batch.append_edges(new_edge_attr, new_edges, idx)
        assert batch.e.shape[0] == n_edges + 20
        assert batch.edge_idx.shape[0] == n_edges + 20
        assert batch.edges.shape[1] == n_edges + 20

        assert getattr(batch, attr).requires_grad




class TestInvalidGraphData:
    def test_invalid_number_of_edges(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(1, 3),
                torch.randint(0, 10, torch.Size([2, 6]))
            )

    def test_invalid_number_of_nodes(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(1, 3),
                torch.randint(11, 12, torch.Size([2, 6]))
            )

    def test_invalid_number_of_edges(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10),
                torch.randn(5, 4),
                torch.randn(1, 3),
                torch.randint(0, 10, torch.Size([2, 5]))
            )

    def test_invalid_global_shape(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(3),
                torch.randint(11, 12, torch.Size([2, 6]))
            )

    def test_invalid_n_edges(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(1, 3),
                torch.randint(0, 10, torch.Size([3, 5]))
            )

    def test_invalid_edge_ndims(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5),
                torch.randn(1, 3),
                torch.randint(0, 10, torch.Size([2, 5]))
            )

    def test_invalid_global_ndims(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(1),
                torch.randint(0, 10, torch.Size([2, 5]))
            )


def test_random_data():
    random_data(5, 4, 3)


class TestGraphBatch:

    def test_basic_batch(self):
        data1 = GraphData(
            torch.randn(10, 10),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3]))
        )

        data2 = GraphData(
            torch.randn(12, 10),
            torch.randn(4, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 4]))
        )

        batch = GraphBatch.from_data_list([data1, data2])
        assert batch.x.shape[0] == 22
        assert batch.e.shape[0] == 7
        assert batch.edges.shape[1] == 7
        assert batch.g.shape[0] == 2
        assert torch.all(torch.eq(batch.node_idx, torch.tensor([0] * 10 + [1] * 12)))
        assert torch.all(torch.eq(batch.edge_idx, torch.tensor([0] * 3 + [1] * 4)))


    def test_basic_batch2(self):

        data1 = GraphData(
            torch.tensor([
                [0],
                [0]
            ]),
            torch.tensor([
                [0],
                [0]
            ]),
            torch.tensor([
                [0]
            ]),
            torch.tensor([
                [0, 1],
                [1, 0]
            ])
        )

        data2 = GraphData(
            torch.tensor([
                [0],
                [0],
                [0],
                [0],
                [0]
            ]),
            torch.tensor([
                [0],
                [0],
                [0]
            ]),
            torch.tensor([
                [0]
            ]),
            torch.tensor([
                [1, 2, 1],
                [4, 2, 1]
            ])
        )

        batch = GraphBatch.from_data_list([data1, data2])
        print(batch.edges)

        datalist2 = batch.to_data_list()
        print(datalist2[0].edges)
        print(datalist2[1].edges)



    @flaky(max_runs=20, min_passes=20)
    def test_to_and_from_datalist(self):
        data1 = GraphData(
            torch.randn(4, 2),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 4, torch.Size([2, 3]))
        )

        data2 = GraphData(
            torch.randn(2, 2),
            torch.randn(4, 4),
            torch.randn(1, 3),
            torch.randint(0, 2, torch.Size([2, 4]))
        )

        batch = GraphBatch.from_data_list([data1, data2])

        datalist2 = batch.to_data_list()

        print(data1.x)

        print(datalist2[0].x)

        print(data2.x)
        print(datalist2[1].x)

        print(data1.edges)
        print(data2.edges)
        print(datalist2[0].edges)
        print(datalist2[1].edges)

        for d1, d2 in zip([data1, data2], datalist2):
            assert torch.allclose(d1.x, d2.x)
            assert torch.allclose(d1.e, d2.e)
            assert torch.allclose(d1.g, d2.g)
            assert torch.all(torch.eq(d1.edges, d2.edges))
            assert d1.allclose(d2)

    @pytest.mark.parametrize('offsets', [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1)
    ])
    def test_invalid_batch(self, offsets):
        data1 = GraphData(
            torch.randn(10, 10),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3]))
        )

        data2 = GraphData(
            torch.randn(12, 10 + offsets[0]),
            torch.randn(4, 4 + offsets[1]),
            torch.randn(1, 3 + offsets[2]),
            torch.randint(0, 10, torch.Size([2, 4]))
        )

        with pytest.raises(RuntimeError):
            GraphBatch.from_data_list([data1, data2])

    @pytest.mark.parametrize('n', [3, 10, 1000])
    def test_from_data_list(self, n):
        datalist = [random_data(5, 3, 4) for _ in range(n)]
        batch = GraphBatch.from_data_list(datalist)
        assert batch.x.shape[0] > n
        assert batch.e.shape[0] > n
        assert batch.g.shape[0] == n
        assert batch.x.shape[1] == 5
        assert batch.e.shape[1] == 3
        assert batch.g.shape[1] == 4

    def test_to_datalist(self):
        datalist = [random_data(5, 6, 7) for _ in range(1000)]
        batch = GraphBatch.from_data_list(datalist)
        print(batch.shape)
        print(batch.size)
        datalist2 = batch.to_data_list()

        assert len(datalist) == len(datalist2)

        def sort(a):
            return a[:, torch.sort(a).indices[0]]

        for data in datalist2:
            print(sort(data.edges))

        for data in datalist2:
            print(sort(data.edges))

        for d1, d2 in zip(datalist, datalist2):
            assert d1.allclose(d2)

    @pytest.mark.parametrize(
        'fkey_gkey', [
            ('features', 'data'),
            ('myfeatures', 'mydata')
        ]
    )
    def test_to_networkx_list(self, fkey_gkey):
        fkey, gkey = fkey_gkey
        datalist = [random_data(5, 5, 5) for _ in range(3)]
        batch = GraphBatch.from_data_list(datalist)
        graphs = batch.to_networkx_list(feature_key=fkey, global_attr_key=gkey)
        for data, graph in zip(datalist, graphs):
            Comparator.data_to_nx(data, graph, fkey, gkey)

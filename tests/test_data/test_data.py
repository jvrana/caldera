from pyrographnets.data.graph_data import GraphData
import torch
import pytest
import networkx as nx


class Comparator:

    @staticmethod
    def data_to_nx(data, g, fkey, gkey):
        """Compare `GraphData` to `nx.DiGraph` instance."""
        assert data.x.shape[0] == g.number_of_nodes()

        assert data.e.shape[0] == g.number_of_edges()

        assert data.edges.shape[1] == g.number_of_edges()

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

        # check global data
        assert torch.all(torch.eq(gdata, data.g))
        assert gdata is not data.g

        # check node data
        nodes = list(g.nodes(data=True))
        for i in range(len(nodes)):
            assert torch.all(torch.eq(nodes[i][1][fkey], data.x[i]))

        # check edge data
        edges = list(g.edges(data=True))
        for i in range(len(edges)):
            assert torch.all(torch.eq(edges[i][2][fkey], data.e[i]))


class TestGraphDataConstructor:

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
        assert isinstance(g, nx.DiGraph)
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

        g = nx.DiGraph()
        g.add_node('node1', **{fkey: torch.randn(5)})
        g.add_node('node2', **{fkey: torch.randn(5)})
        g.add_edge('node1', 'node2', **{fkey: torch.randn(4)})

        setattr(g, gkey, {fkey: torch.randn(3)})

        data = GraphData.from_networkx(g, **kwargs)

        Comparator.data_to_nx(data, g, fkey, gkey)


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
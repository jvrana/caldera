import networkx as nx
import numpy as np
import pytest
import torch
from flaky import flaky

from caldera.data import GraphBatch
from caldera.data import GraphData

random_graph_data = GraphData.random


def test_random_data():
    random_graph_data(5, 4, 3)


def rndm_data():
    return pytest.mark.parametrize(
        "random_data_example",
        [
            (GraphData, (5, 4, 3)),
            (GraphData, (5, 7, 10)),
            (GraphBatch, (10, 5, 6, 7)),
            (GraphBatch, (100, 5, 6, 7)),
        ],
        indirect=True,
        ids=lambda x: str(x),
    )


def test_random_data_example(random_data_example):
    print(random_data_example)


class TestRandomize:
    def test_randomize_data(self):
        data = GraphData.random(5, 4, 3)
        x, e, g = data.x.clone(), data.e.clone(), data.g.clone()
        data.randomize_()
        x2, e2, g2 = data.x.clone(), data.e.clone(), data.g.clone()
        assert not torch.allclose(x, x2)
        assert not torch.allclose(e, e2)
        assert not torch.allclose(g, g2)

    def test_randomize_batch(self):
        data = GraphBatch.random_batch(10, 5, 4, 3)
        x, e, g = data.x.clone(), data.e.clone(), data.g.clone()
        data.randomize_()
        x2, e2, g2 = data.x.clone(), data.e.clone(), data.g.clone()
        assert not torch.allclose(x, x2)
        assert not torch.allclose(e, e2)
        assert not torch.allclose(g, g2)

    def test_randomize_new_shapes(self):
        data = GraphBatch.random_batch(10, 5, 4, 3)
        data.randomize_(1, 2, 3)
        x2, e2, g2 = data.x.clone(), data.e.clone(), data.g.clone()
        assert x2.shape[1] == 1
        assert e2.shape[1] == 2
        assert g2.shape[1] == 3


class Comparator:
    @staticmethod
    def data_to_nx(data, g, fkey, gkey):
        """Compare `GraphData` to `nx.DiGraph` instance."""
        assert data.x.shape[0] == g.number_of_nodes(), "Check number of nodes"

        assert data.e.shape[0] == g.number_of_edges(), "Check number of edges"

        assert data.edges.shape[1] == g.number_of_edges(), "Check edges vs edge attr"

        # ensure feature key is in node data
        for _, ndata in g.nodes(data=True):
            assert fkey in ndata

        # ensure feature key is in edge data
        for _, _, edata in g.edges(data=True):
            assert fkey in edata

        # ensure feature key is in global data
        gdata = torch.tensor(g.get_global(gkey)[fkey], dtype=torch.float)
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
            expected_node = torch.tensor(nodes[i][1][fkey], dtype=torch.float)
            assert torch.all(torch.eq(expected_node, data.x[i]))

        # check edge data
        for i, (n1, n2, ne) in enumerate(g.ordered_edges):
            edata = g[n1][n2][ne]
            expected_edge = torch.tensor(edata[fkey], dtype=torch.float)
            assert torch.all(torch.eq(expected_edge, data.e[i]))


# TODO: test data for numpy or torch
class TestGraphData:
    def test_graph_data_init_0(self):
        data = GraphData(
            torch.randn(10, 5),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3])),
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
            torch.randint(0, 10, torch.Size([2, 5])),
        )
        assert data.x.shape == torch.Size([10, 5])
        assert data.e.shape == torch.Size([5, 4])
        assert data.edges.shape == torch.Size([2, 5])
        assert data.g.shape == torch.Size([1, 3])

    @pytest.mark.parametrize(
        "keys", [(None, None), ("myfeatures", "mydata"), ("features", "data")]
    )
    def test_to_networkx(self, keys):
        kwargs = {"feature_key": "features", "global_attr_key": "data"}
        feature_key, global_attr_key = keys
        if feature_key is not None:
            kwargs["feature_key"] = feature_key
        else:
            del kwargs["feature_key"]
        if global_attr_key is not None:
            kwargs["global_attr_key"] = global_attr_key
        else:
            del kwargs["global_attr_key"]

        data = GraphData(
            torch.randn(10, 5),
            torch.randn(5, 4),
            torch.randn(1, 3),
            torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]),
        )

        g = data.to_networkx(**kwargs)
        assert isinstance(g, nx.OrderedMultiDiGraph)
        assert g.number_of_nodes() == 10
        assert g.number_of_edges() == 5

        fkey = kwargs.get("feature_key", "features")
        gkey = kwargs.get("global_attr_key", None)

        Comparator.data_to_nx(data, g, fkey, gkey)

    @pytest.mark.parametrize(
        "keys", [(None, None), ("myfeatures", "mydata"), ("features", "data")]
    )
    def test_from_networkx(self, keys):
        kwargs = {"feature_key": "features", "global_attr_key": "data"}
        feature_key, global_attr_key = keys
        if feature_key is not None:
            kwargs["feature_key"] = feature_key
        else:
            del kwargs["feature_key"]
        if global_attr_key is not None:
            kwargs["global_attr_key"] = global_attr_key
        else:
            del kwargs["global_attr_key"]

        fkey = kwargs.get("feature_key", "features")
        gkey = kwargs.get("global_attr_key", global_attr_key)

        g = nx.OrderedMultiDiGraph()
        g.add_node("node1", **{fkey: np.random.randn(5)})
        g.add_node("node2", **{fkey: np.random.randn(5)})
        g.add_edge("node1", "node2", **{fkey: np.random.randn(4)})
        g.ordered_edges = [("node1", "node2", 0)]
        g.set_global({fkey: np.random.randn(3)}, gkey)

        data = GraphData.from_networkx(g, **kwargs)

        Comparator.data_to_nx(data, g, fkey, gkey)

    @pytest.mark.parametrize(
        "cls",
        [
            nx.DiGraph,
            pytest.param(
                nx.Graph, marks=pytest.mark.xfail(raises=TypeError, strict=True)
            ),
        ],
    )
    def test_from_networkx_types(self, cls):
        g = cls()
        GraphData.from_networkx(g)

    def test_empty_networkx(self):
        """Empty graphs should be OK."""
        g = nx.DiGraph()
        GraphData.from_networkx(g)

    def test_from_networkx_missing_node_data(self):
        g = nx.DiGraph()
        g.add_node(1)
        g.add_edge(1, 2, feature=np.array([10.0]))
        g.set_global({"feature": np.array([12.0])})
        data = GraphData.from_networkx(g, feature_key="feature")
        assert data.x.shape == (2, 0)
        assert torch.all(data.e == torch.tensor([[10.0]]))
        assert torch.all(data.g == torch.tensor([[12.0]]))

    def test_from_networkx_missing_edge_data(self):
        g = nx.DiGraph()
        g.add_node(1, feature=np.array([10.0]))
        g.add_node(2, feature=np.array([11.0]))
        g.add_edge(1, 2)
        g.set_global({"feature": np.array([12.0])})
        data = GraphData.from_networkx(g, feature_key="feature")
        assert data.e.shape == (1, 0)
        assert torch.all(data.x == torch.tensor([[10.0], [11.0]]))
        assert torch.all(data.g == torch.tensor([[12.0]]))

    def test_from_networkx_missing_glob_data(self):
        g = nx.DiGraph()
        g.add_node(1, feature=np.array([10.0]))
        g.add_node(2, feature=np.array([11.0]))
        g.add_edge(1, 2, feature=np.array([12.0]))
        data = GraphData.from_networkx(g, feature_key="feature")
        assert torch.all(data.x == torch.tensor([[10.0], [11.0]]))
        assert torch.all(data.e == torch.tensor([[12.0]]))
        assert data.g.shape == (1, 0)

    def test_from_networkx_different_node_shapes(self):
        g = nx.DiGraph()
        g.add_node(1, feature=np.array([[10.0, 11.0]]))
        g.add_node(2, feature=np.array([[10.0], [11.0], [12.0]]))
        with pytest.raises(RuntimeError):
            GraphData.from_networkx(g, feature_key="feature")

    def test_from_networkx_different_edge_shapes(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, feature=np.array([[10.0, 11.0]]))
        g.add_edge(2, 3, feature=np.array([[10.0], [11.0]]))
        with pytest.raises(RuntimeError):
            GraphData.from_networkx(g, feature_key="feature")

    @pytest.mark.parametrize(
        "keys", [(None, None), ("myfeatures", "mydata"), ("features", "data")]
    )
    def test_from_networkx_no_edge(self, keys):
        kwargs = {"feature_key": "features", "global_attr_key": "data"}
        feature_key, global_attr_key = keys
        if feature_key is not None:
            kwargs["feature_key"] = feature_key
        else:
            del kwargs["feature_key"]
        if global_attr_key is not None:
            kwargs["global_attr_key"] = global_attr_key
        else:
            del kwargs["global_attr_key"]

        fkey = kwargs.get("feature_key", "features")
        gkey = kwargs.get("global_attr_key", None)

        g = nx.OrderedMultiDiGraph()
        g.add_node("node1", **{fkey: np.random.randn(5)})
        g.add_node("node2", **{fkey: np.random.randn(5)})
        g.ordered_edges = []
        # g.add_edge('node1', 'node2', **{fkey: torch.randn(4)})
        g.set_global({fkey: np.random.randn(3)}, gkey)

        data = GraphData.from_networkx(g, **kwargs)

        Comparator.data_to_nx(data, g, fkey, gkey)


# @rndm_data()
class TestApply:
    def test_apply_(self, random_data_example):
        data = random_data_example
        data2 = data.apply_(lambda x: x.contiguous)
        assert id(data2) == id(data), "`apply` should return the same instance"

    def test_to_cpu_does_share(self, random_data_example):
        data = random_data_example
        data2 = data.apply(lambda x: x.cpu())
        assert id(data) != id(data2), "`apply` should return a new instance"
        assert data.share_storage(data2), "apply `cpu()` should share the same storage"

    def test_to_gpu_does_not_share(self, random_data_example):
        data = random_data_example
        if torch.cuda.is_available():
            device = "cuda:" + str(torch.cuda.current_device())
            data2 = data.apply(lambda x: x.to(device))
            assert not data.share_storage(
                data2
            ), "apply `gpu()` should not share the same storage"
            assert not data2.share_storage(
                data
            ), "apply `gpu()` should not share the same storage"
            for k, v in data._tensors:
                assert v.device.type == "cpu"
            for k, v in data2._tensors:
                assert v.device.type == "cuda"

    def test_to_cuda(self, random_data_example):
        data = random_data_example
        if torch.cuda.is_available():
            device = "cuda:" + str(torch.cuda.current_device())
            data2 = data.to(device)
            assert id(data2) != id(data)
            assert not data.share_storage(data2)
            assert not data2.share_storage(data)
            for k, v in data._tensors:
                assert v.device.type == "cpu"
            for k, v in data2._tensors:
                assert v.device.type == "cuda"

    @pytest.mark.parametrize("req", [True, False])
    def tests_seq_requires_grad(self, random_data_example, req):
        data = random_data_example

        req1, req2 = req, (not req)

        for req in [req1, req2]:
            data.requires_grad = req
            assert data.requires_grad is req
            for k in data._differentiable:
                v = getattr(data, k)
                assert v.requires_grad is req


# TODO: TestComparison
class TestComparison:
    def test_eq(self):
        args1 = (
            torch.randn(20, 5),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3])),
        )
        args2 = (args1[0][:], args1[1][:], args1[2][:], args1[3][:])
        data1 = GraphData(*args1)
        data2 = GraphData(*args2)
        assert data1 == data2
        assert not id(data1) == id(data2)

    def test_not_eq(self):
        args1 = (
            torch.randn(20, 5),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3])),
        )
        args2 = (args1[0][:10], args1[1][:], args1[2][:], args1[3][:])
        data1 = GraphData(*args1)
        data2 = GraphData(*args2)
        assert not data1 == data2
        assert not id(data1) == id(data2)

    def test_does_share_storage(self):
        args1 = (
            torch.randn(20, 5),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3])),
        )
        args2 = (args1[0][:10], args1[1][:], args1[2][:], args1[3][:])
        data1 = GraphData(*args1)
        data2 = GraphData(*args2)
        assert data1.share_storage(data2)
        assert data2.share_storage(data1)

    def test_does_not_share_storage(self):
        args1 = (
            torch.randn(20, 5),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3])),
        )
        args2 = (args1[0][:10], args1[1][:], args1[2][:], args1[3][:])
        args2 = (torch.tensor(x) for x in args2)
        data1 = GraphData(*args1)
        data2 = GraphData(*args2)
        assert not data1.share_storage(data2)
        assert not data2.share_storage(data1)


class TestGraphDataModifiers:
    def test_append_nodes(self):
        data = GraphData(
            torch.randn(10, 5),
            torch.randn(5, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 5])),
        )

        assert data.x.shape[0] == 10
        data.append_nodes(torch.randn(2, 5))
        assert data.x.shape[0] == 12

    def test_append_edges(self):
        data = GraphData(
            torch.randn(10, 5),
            torch.randn(5, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 5])),
        )

        e = torch.randn(3, 4)
        edges = torch.randint(0, 10, torch.Size([2, 3]))
        data.append_edges(e, edges)

    def test_invalid_append_edges(self):
        data = GraphData(
            torch.randn(10, 5),
            torch.randn(5, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 5])),
        )

        e = torch.randn(3, 4)
        edges = torch.randint(0, 10, torch.Size([2, 4]))
        with pytest.raises(RuntimeError):
            data.append_edges(e, edges)

    def test_differentiable__mask_connectivity(self):
        """Tests changing the connectivity of the batch."""
        data = random_graph_data(5, 4, 3)
        print(data.size)


class TestGraphBatchModifiers:
    def test_batch_append_nodes(self):

        datalist = [random_graph_data(5, 6, 7) for _ in range(10)]
        batch = GraphBatch.from_data_list(datalist)

        x = torch.randn(3, 5)
        idx = torch.tensor([0, 1, 2])

        node_shape = batch.x.shape
        batch.append_nodes(x, idx)
        print(batch.node_idx.shape)

        assert node_shape[0] < batch.x.shape[0]
        print(batch.node_idx.shape)
        print(batch.x.shape)

    @pytest.mark.parametrize("attr", ["x", "e", "g"])
    def test_is_differentiable__to_datalist(self, attr):
        datalist = [random_graph_data(5, 3, 4) for _ in range(300)]

        batch = GraphBatch.from_data_list(datalist)

        getattr(batch, attr).requires_grad = True

        datalist = batch.to_data_list()
        for data in datalist:
            assert getattr(data, attr).requires_grad

    @pytest.mark.parametrize("attr", ["x", "e", "g"])
    def test_is_differentiable__from_datalist(self, attr):
        datalist = [random_graph_data(5, 3, 4) for _ in range(300)]
        for data in datalist:
            getattr(data, attr).requires_grad = True
        batch = GraphBatch.from_data_list(datalist)
        assert getattr(batch, attr).requires_grad

    @pytest.mark.parametrize("attr", ["x", "e", "g"])
    def test_is_differentiable__append_nodes(self, attr):
        datalist = [random_graph_data(5, 3, 4) for _ in range(300)]
        for data in datalist:
            getattr(data, attr).requires_grad = True
        batch = GraphBatch.from_data_list(datalist)

        new_nodes = torch.randn(10, 5)
        idx = torch.ones(10, dtype=torch.long)
        n_nodes = batch.x.shape[0]
        batch.append_nodes(new_nodes, idx)
        assert batch.x.shape[0] == n_nodes + 10
        assert getattr(batch, attr).requires_grad

    @pytest.mark.parametrize("attr", ["x", "e", "g"])
    def test_is_differentiable__append_edges(self, attr):
        datalist = [random_graph_data(5, 3, 4) for _ in range(300)]
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
    def test_invalid_number_of_edges(self, cls):
        with pytest.raises(RuntimeError):
            cls(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(1, 3),
                torch.randint(0, 10, torch.Size([2, 6])),
            )

    def test_invalid_number_of_nodes(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(1, 3),
                torch.randint(11, 12, torch.Size([2, 6])),
            )

    def test_invalid_number_of_edges(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10),
                torch.randn(5, 4),
                torch.randn(1, 3),
                torch.randint(0, 10, torch.Size([2, 5])),
            )

    def test_invalid_global_shape(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(3),
                torch.randint(11, 12, torch.Size([2, 6])),
            )

    def test_invalid_n_edges(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(1, 3),
                torch.randint(0, 10, torch.Size([3, 5])),
            )

    def test_invalid_edge_ndims(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5),
                torch.randn(1, 3),
                torch.randint(0, 10, torch.Size([2, 5])),
            )

    def test_invalid_global_ndims(self):
        with pytest.raises(RuntimeError):
            GraphData(
                torch.randn(10, 5),
                torch.randn(5, 4),
                torch.randn(1),
                torch.randint(0, 10, torch.Size([2, 5])),
            )


class TestGraphBatch:
    def test_basic_batch(self):
        data1 = GraphData(
            torch.randn(10, 10),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3])),
        )

        data2 = GraphData(
            torch.randn(12, 10),
            torch.randn(4, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 4])),
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
            torch.tensor([[0], [0]]),
            torch.tensor([[0], [0]]),
            torch.tensor([[0]]),
            torch.tensor([[0, 1], [1, 0]]),
        )

        data2 = GraphData(
            torch.tensor([[0], [0], [0], [0], [0]]),
            torch.tensor([[0], [0], [0]]),
            torch.tensor([[0]]),
            torch.tensor([[1, 2, 1], [4, 2, 1]]),
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
            torch.randint(0, 4, torch.Size([2, 3])),
        )

        data2 = GraphData(
            torch.randn(2, 2),
            torch.randn(4, 4),
            torch.randn(1, 3),
            torch.randint(0, 2, torch.Size([2, 4])),
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

    @pytest.mark.parametrize("offsets", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    def test_invalid_batch(self, offsets):
        data1 = GraphData(
            torch.randn(10, 10),
            torch.randn(3, 4),
            torch.randn(1, 3),
            torch.randint(0, 10, torch.Size([2, 3])),
        )

        data2 = GraphData(
            torch.randn(12, 10 + offsets[0]),
            torch.randn(4, 4 + offsets[1]),
            torch.randn(1, 3 + offsets[2]),
            torch.randint(0, 10, torch.Size([2, 4])),
        )

        with pytest.raises(RuntimeError):
            GraphBatch.from_data_list([data1, data2])

    @pytest.mark.parametrize("n", [3, 10, 1000])
    def test_from_data_list(self, n):
        datalist = [random_graph_data(5, 3, 4) for _ in range(n)]
        batch = GraphBatch.from_data_list(datalist)
        assert batch.x.shape[0] > n
        assert batch.e.shape[0] > n
        assert batch.g.shape[0] == n
        assert batch.x.shape[1] == 5
        assert batch.e.shape[1] == 3
        assert batch.g.shape[1] == 4

    def test_to_datalist(self):
        datalist = [random_graph_data(5, 6, 7) for _ in range(1000)]
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
        "fkey_gkey", [("features", "data"), ("myfeatures", "mydata")]
    )
    def test_to_networkx_list(self, fkey_gkey):
        fkey, gkey = fkey_gkey
        datalist = [random_graph_data(5, 5, 5) for _ in range(3)]
        batch = GraphBatch.from_data_list(datalist)
        graphs = batch.to_networkx_list(feature_key=fkey, global_attr_key=gkey)
        for data, graph in zip(datalist, graphs):
            Comparator.data_to_nx(data, graph, fkey, gkey)


def test_graph_data_random():
    assert GraphData.random(5, 5, 5)


def test_graph_batch_random_batch():
    batch = GraphBatch.random_batch(10, 5, 5, 5)
    print(batch.size)
    print(batch.shape)


# @rndm_data()
class TestCloneCopy:
    def test_copy(self, random_data_example):
        data = random_data_example
        data2 = data.copy()
        assert id(data) != id(data2)
        c = data.share_storage(data2, return_dict=True)
        print(data.size)
        print(c)
        assert not data.share_storage(data2)

    def test_clone(self, random_data_example):
        data = random_data_example
        data2 = data.clone()
        assert id(data) != id(data2)
        assert not data.share_storage(data2)


class TestView:
    @pytest.mark.parametrize(
        "slices",
        [
            (None, None, None),
            (slice(None, 2), None, None),
            (None, slice(None, 3), None),
            (None, None, slice(None, 3)),
        ],
    )
    def test_view_graph_data(self, slices):
        data = GraphData.random(5, 5, 5, min_nodes=10, max_nodes=10)
        assert data.shape == (5, 5, 5)
        data_view = data.view(*slices)
        print(data_view.shape)
        # assert data_view.shape == (5, 3, 5)
        assert data_view.share_storage(data)


class TestCudaData:
    def test_to_datalist(self, device):
        batch = GraphBatch.random_batch(2, 5, 4, 3)
        batch = batch.to(device)
        datalist = batch.to_data_list()

    def test_from_datalist(self, device):
        batch = GraphBatch.random_batch(2, 5, 4, 3)
        batch = batch.to(device)
        datalist = batch.to_data_list()
        GraphBatch.from_data_list(datalist)


def test_():

    data1 = GraphData(
        torch.randn(1, 5),
        torch.randn(1, 4),
        torch.randn(1, 3),
        torch.LongTensor([[0], [0]]),
    )

    data2 = GraphData(
        torch.randn(1, 5),
        torch.randn(0, 4),
        torch.randn(1, 3),
        torch.LongTensor([[], []]),
    )

    batch = GraphBatch.from_data_list([data1, data2])


def test_cat_graph_batch():
    b1 = GraphBatch.random_batch(10, 5, 4, 3)
    b2 = GraphBatch(
        torch.rand_like(b1.x),
        torch.rand_like(b1.e),
        torch.rand_like(b1.g),
        edges=b1.edges,
        node_idx=b1.node_idx,
        edge_idx=b1.edge_idx,
    )
    b3 = b1.cat(b2)
    assert b3.shape == (10, 8, 6)
    assert torch.allclose(b3.x, torch.cat([b1.x, b2.x], 1))

    b4 = GraphBatch.cat(b1, b2)
    assert b4.shape == (10, 8, 6)
    assert torch.allclose(b4.x, torch.cat([b1.x, b2.x], 1))


@pytest.mark.parametrize("zero_out", [None, "edge_idx", "node_idx", "edges"])
def test_same_topology(zero_out):
    b1 = GraphBatch.random_batch(10, 5, 4, 3)
    b2 = GraphBatch(
        torch.rand_like(b1.x),
        torch.rand_like(b1.e),
        torch.rand_like(b1.g),
        edges=b1.edges,
        node_idx=b1.node_idx,
        edge_idx=b1.edge_idx,
    )

    if zero_out:
        setattr(b2, zero_out, torch.zeros_like(getattr(b2, zero_out)))

    if zero_out:
        assert not b1.same_topology(b2)
        assert not b1.same_topology(b2, b1, b2)
        assert not GraphBatch.same_topology(b1, b2)
    else:
        assert b1.same_topology(b2)
        assert b1.same_topology(b2, b1, b2)
        assert GraphBatch.same_topology(b1, b2)


class TestDataOperators():

    def test_sum(self):
        a = GraphData.random(5, 4, 3)
        b = a.clone().randomize_()
        c = GraphData.add(a, b)
        assert torch.allclose(a.x + b.x, c.x)
        assert torch.allclose(a.e + b.e, c.e)
        assert torch.allclose(a.g + b.g, c.g)

    def test_add(self):
        a = GraphData.random(5, 4, 3)
        b = a.clone().randomize_()
        c = a + b
        assert torch.allclose(a.x + b.x, c.x)
        assert torch.allclose(a.e + b.e, c.e)
        assert torch.allclose(a.g + b.g, c.g)

    def test_add_scalar(self):
        a = GraphData.random(5, 4, 3)
        c = a + 2
        assert torch.allclose(a.x + 2, c.x)
        assert torch.allclose(a.e + 2, c.e)
        assert torch.allclose(a.g + 2, c.g)

    def test_add_tensor(self):
        a = GraphData.random(5, 4, 3)
        c = a + 2
        assert torch.allclose(a.x + 2, c.x)
        assert torch.allclose(a.e + 2, c.e)
        assert torch.allclose(a.g + 2, c.g)

    def test_mul_scalar(self):
        a = GraphData.random(5, 4, 3)
        c = a * 2
        assert torch.allclose(a.x * 2, c.x)
        assert torch.allclose(a.e * 2, c.e)
        assert torch.allclose(a.g * 2, c.g)

    def test_mul_tensor(self):
        a = GraphData.random(5, 4, 3)
        c = a * torch.tensor(2.)
        assert torch.allclose(a.x * 2, c.x)
        assert torch.allclose(a.e * 2, c.e)
        assert torch.allclose(a.g * 2, c.g)


    def test_mul(self):
        a = GraphData.random(5, 4, 3)
        b = a.clone().randomize_()
        c = GraphData.prod(a, b)
        assert torch.allclose(a.x * b.x, c.x)
        assert torch.allclose(a.e * b.e, c.e)
        assert torch.allclose(a.g * b.g, c.g)

    def test_mul_magic(self):
        a = GraphData.random(5, 4, 3)
        b = a.clone().randomize_()
        c = a * b
        assert torch.allclose(a.x * b.x, c.x)
        assert torch.allclose(a.e * b.e, c.e)
        assert torch.allclose(a.g * b.g, c.g)

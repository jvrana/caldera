import networkx as nx


def test_global_access():
    g = nx.OrderedMultiDiGraph()
    assert g.get_global() == {}
    g.get_global().update({"u": 2})
    g.get_global()["x"] = 1
    assert g.get_global()["x"] == 1


def test_global_access_with_key():
    g = nx.OrderedMultiDiGraph()
    g.data = {}
    assert g.get_global("data") == {}
    g.get_global("data")["k"] = 1
    g.get_global("data").update({"u": 2})
    assert g.get_global("data") == {"k": 1, "u": 2}


def test_get_global_key():
    g = nx.DiGraph()
    print(g.get_global_key())


def test_set_global():
    g = nx.DiGraph()
    g.get_global()["k"] = 2
    g.set_global({"k": 1})
    assert g.get_global() == {"k": 1}

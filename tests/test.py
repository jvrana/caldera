from caldera.data import GraphData
import torch

def test(random_data):

    data = GraphData(
        torch.randn(5, 1),
        torch.randn(5, 1),
        torch.randn(1, 1),
        edges=torch.LongTensor([
            [0, 1],
            [1, 2],
            [1, 3],
            [2, 3],
            [3, 4]
        ]).T
    )

    src, dest = data.edges.tolist()
    edge_dict = {}
    for _src, _dest in zip(src, dest):
        edge_dict.setdefault(_src, set())
        edge_dict[_src].add(_dest)

    nodes = torch.LongTensor([0])

    nlist = nodes.tolist()
    k = 3

    to_visit = set(nlist)
    visited = set()
    discovered = set(nlist)

    i = 0
    while to_visit and i < k:
        i += 1
        v = to_visit.pop()
        visited.add(v)
        if v not in visited:
            discovered.add(v)
        if v in edge_dict:
            neighbors = edge_dict[v]
            to_visit = to_visit.union(neighbors.difference(visited))
            discovered = discovered.union(neighbors)

    print(discovered)




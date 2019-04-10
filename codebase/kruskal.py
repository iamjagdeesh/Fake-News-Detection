parent = dict()
rank = dict()


def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0


def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]


def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
        if rank[root1] == rank[root2]: rank[root2] += 1


def kruskal(graph):
    sum = 0
    for vertice in graph['vertices']:
        make_set(vertice)
        minimum_spanning_tree = set()
        edges = list(graph['edges'])
        #edges.sort()

    edges.sort()
    # print edges
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            sum += weight;
            minimum_spanning_tree.add(edge)

    print(sum)

    return sorted(minimum_spanning_tree)


graph = {
    'vertices': ['1', '2', '3', '4', '5', '6'],
    'edges': set([
        (0, '1', '2'),
        (0, '2', '1'),
        (0, '2', '3'),
        (0, '3', '2'),
        (0, '4', '5'),
        (0, '5', '4'),
        (0, '3', '5'),
        (0, '5', '3'),
        (410, '1', '6'),
        (410, '6', '1'),
        (800, '2', '4'),
        (800, '4', '2')
    ])
}

print(kruskal(graph))
########################################    processing weights data    ##############################################

def path(begin, stop):
    """Provide the starting and end postion names and function will return shortest path"""

    arr= weights

    order = len(arr)
    arr = np.array(arr)
    arr = np.pad(arr, 1, mode="constant")

    adjacency_lists = []
    normal = [1, 2, 3, 4 , 99995, 99999]
    one_ways = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44]
    inf = 9999999999

    # c = 0
    # for i in range(1, order + 1):
    #     for j in range(1, order + 1):
    #         c += 1
    #         names[(j, i)] = c

    for i in range(1, order + 1):
        for j in range(1, order + 1):

            if arr[i][j] in normal:

                if arr[i][j - 1] in one_ways:
                    if arr[i][j - 1] % 10 == 1:  # left one way
                        adjacency_lists.append(
                            (str(names[(i, j)]), str(names[(i, j - 1)]), inf)
                        )
                    else:  # valid left one way
                        adjacency_lists.append(
                            (
                                str(names[(i, j)]),
                                str(names[(i, j - 1)]),
                                arr[i][j - 1] // 10,
                            )
                        )
                elif arr[i][j - 1]:  # normal left
                    adjacency_lists.append(
                        (str(names[(i, j)]), str(names[(i, j - 1)]), arr[i][j - 1])
                    )



                if arr[i][j + 1] in one_ways:
                    if arr[i][j + 1] % 10 == 3:  # right one way
                        adjacency_lists.append(
                            (str(names[(i, j)]), str(names[(i, j + 1)]), inf)
                        )
                    else:  # valid right one way
                        adjacency_lists.append(
                            (
                                str(names[(i, j)]),
                                str(names[(i, j + 1)]),
                                arr[i][j + 1] // 10,
                            )
                        )
                elif arr[i][j + 1]:  # normal right
                    adjacency_lists.append(
                        (str(names[(i, j)]), str(names[(i, j + 1)]), arr[i][j + 1])
                    )



                if arr[i - 1][j] in one_ways:
                    if arr[i - 1][j] % 10 == 2:  # up one way
                        adjacency_lists.append(
                            (str(names[(i, j)]), str(names[(i - 1, j)]), inf)
                        )
                    else:  # valid up one way
                        adjacency_lists.append(
                            (
                                str(names[(i, j)]),
                                str(names[(i - 1, j)]),
                                arr[i - 1][j] // 10,
                            )
                        )
                elif arr[i - 1][j]:  # normal up
                    adjacency_lists.append(
                        (str(names[(i, j)]), str(names[(i - 1, j)]), arr[i - 1][j])
                    )



                if arr[i + 1][j] in one_ways:
                    if arr[i + 1][j] % 10 == 4:  # down one way
                        adjacency_lists.append(
                            (str(names[(i, j)]), str(names[(i + 1, j)]), inf)
                        )
                    else:  # valid down one way
                        adjacency_lists.append(
                            (
                                str(names[(i, j)]),
                                str(names[(i + 1, j)]),
                                arr[i + 1][j] // 10,
                            )
                        )
                elif arr[i + 1][j]:  # normal down
                    adjacency_lists.append(
                        (str(names[(i, j)]), str(names[(i + 1, j)]), arr[i + 1][j])
                    )



            elif arr[i][j] in one_ways:



                if arr[i][j] % 10 == 1:  # left invalid
                    if arr[i][j-1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i, j - 1)]), inf   )  )
                    if arr[i-1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i - 1, j)]), inf   )  )
                    if arr[i+1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i + 1, j)]), inf   )  )

                    if arr[i][j+1] in one_ways: #right one way
                        if arr[i][j + 1] % 10 == 3:  # invalid right one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j + 1)]), inf  )  )
                        else:  # valid rigt one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j + 1)]), arr[i][j + 1] // 10 )  )
                    elif arr[i][j+1] in normal:
                        adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j + 1)]), arr[i][j + 1] )  )



                if arr[i][j] % 10 == 2: #up invalid
                    if arr[i][j-1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i, j - 1)]), inf   )  )
                    if arr[i-1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i - 1, j)]), inf   )  )
                    if arr[i][j+1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i , j + 1)]), inf   )  )

                    if arr[i+1][j] in one_ways: #down one way
                        if arr[i+1][j] % 10 == 4:
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i + 1, j)]), inf  )  )
                        else: #valid down one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i + 1, j)]), arr[i+1][j] // 10  )  )
                    elif arr[i+1][j] in normal:
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i + 1, j)]), arr[i+1][j]  )  )



                if arr[i][j] % 10 == 3:  # right invalid
                    if arr[i-1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i - 1, j)]), inf   )  )
                    if arr[i+1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i + 1, j)]), inf   )  )
                    if arr[i][j+1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i , j + 1)]), inf   )  )

                    if arr[i][j-1] in one_ways: #left one way
                        if arr[i][j - 1] % 10 == 1:  # invalid left one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j - 1)]), inf  )  )
                        else:  # valid rigt one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j - 1)]), arr[i][j - 1] // 10 )  )
                    elif arr[i][j-1] in normal:
                        adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j - 1)]), arr[i][j - 1] )  )



                if arr[i][j] % 10 == 4: #down invalid
                    if arr[i][j-1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i, j - 1)]), inf   )  )
                    if arr[i+1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i + 1, j)]), inf   )  )
                    if arr[i][j+1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i , j + 1)]), inf   )  )

                    if arr[i-1][j] in one_ways: #up one way
                        if arr[i-1][j] % 10 == 2:
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i - 1, j)]), inf  )  )
                        else: #valid up one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i - 1, j)]), arr[i-1][j] // 10  )  )
                    elif arr[i-1][j] in normal:
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i - 1, j)]), arr[i-1][j]  )  )




    # print(arr)
    arr = arr[1:-1, 1:-1]
    # print(arr)

    print("Block numbers : ",names)

##################################################    path planning    #########################################################

    inf = float('inf')
    Edge = namedtuple('Edge', 'start, end, cost')


    def make_edge(start, end, cost=1):
        return Edge(start, end, cost)


    class Graph:
        def __init__(self, edges):
            # let's check that the data is right
            wrong_edges = [i for i in edges if len(i) not in [2, 3]]
            if wrong_edges:
                raise ValueError('Wrong edges data: {}'.format(wrong_edges))

            self.edges = [make_edge(*edge) for edge in edges]

        @property
        def vertices(self):
            return set(
                sum(
                    ([edge.start, edge.end] for edge in self.edges), []
                )
            )

        def get_node_pairs(self, n1, n2, both_ends=True):
            if both_ends:
                node_pairs = [[n1, n2], [n2, n1]]
            else:
                node_pairs = [[n1, n2]]
            return node_pairs

        def remove_edge(self, n1, n2, both_ends=True):
            node_pairs = self.get_node_pairs(n1, n2, both_ends)
            edges = self.edges[:]
            for edge in edges:
                if [edge.start, edge.end] in node_pairs:
                    self.edges.remove(edge)

        def add_edge(self, n1, n2, cost=1, both_ends=True):
            node_pairs = self.get_node_pairs(n1, n2, both_ends)
            for edge in self.edges:
                if [edge.start, edge.end] in node_pairs:
                    return ValueError('Edge {} {} already exists'.format(n1, n2))

            self.edges.append(Edge(start=n1, end=n2, cost=cost))
            if both_ends:
                self.edges.append(Edge(start=n2, end=n1, cost=cost))

        @property
        def neighbours(self):
            neighbours = {vertex: set() for vertex in self.vertices}
            for edge in self.edges:
                neighbours[edge.start].add((edge.end, edge.cost))

            return neighbours

        def dijkstra(self, source, dest):
            assert source in self.vertices, 'Such source node doesn\'t exist'
            distances = {vertex: inf for vertex in self.vertices}
            previous_vertices = {
                vertex: None for vertex in self.vertices
            }
            distances[source] = 0
            vertices = self.vertices.copy()

            while vertices:
                current_vertex = min(
                    vertices, key=lambda vertex: distances[vertex])
                vertices.remove(current_vertex)
                if distances[current_vertex] == inf:
                    break
                for neighbour, cost in self.neighbours[current_vertex]:
                    alternative_route = distances[current_vertex] + cost
                    if alternative_route < distances[neighbour]:
                        distances[neighbour] = alternative_route
                        previous_vertices[neighbour] = current_vertex

            path, current_vertex = deque(), dest
            while previous_vertices[current_vertex] is not None:
                path.appendleft(current_vertex)
                current_vertex = previous_vertices[current_vertex]
            if path:
                path.appendleft(current_vertex)
            return path


    graph = Graph(adjacency_lists)

    return list(graph.dijkstra(str(begin), str(stop)))
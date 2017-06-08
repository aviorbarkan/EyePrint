import sys
import heapq
import numpy as np


class Vertex:
    def __init__(self, node):
        self.id = node
        self.neighbors = []
        self.adjacent = {}
        # Set distance to infinity for all nodes
        self.distance = sys.maxsize
        # Mark all nodes unvisited
        self.visited = False
        # Predecessor
        self.previous = None
        self.num_of_nodes = 0

    def add_neighbor(self, neighbor, weight=0):
        self.neighbors.append(neighbor.id)
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.distance == other.distance
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.distance < other.distance
        return NotImplemented

    def __hash__(self):
        return id(self)

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])


class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost=0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        # self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous


def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return


def dijkstra(aGraph, start):
    print('''Dijkstra's shortest path''')
    # Set the distance for the start node to zero
    start.set_distance(0)
    penalty = aGraph.num_vertices ** 2

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.get_distance(), v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        # for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)
            # + (current.num_of_nodes * penalty)

            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
                next.num_of_nodes = current.num_of_nodes + 1
                print('updated : current = %s next = %s new_dist = %s' % (current.get_id(),
                                                                          next.get_id(),
                                                                          next.get_distance()))
            else:
                print('not updated : current = %s next = %s new_dist = %s' % (current.get_id(),
                                                                              next.get_id(),
                                                                              next.get_distance()))

        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(), v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)


def shortest_path(graph, src_node, target_node, path_length):
    # TODO: optimize for multiple runnins of the same graph but different divisions
    v = graph.num_vertices
    if path_length == 0:
        if src_node.id == target_node.id:
            return [src_node.id, target_node.id]
        else:
            return np.inf

    weights = np.ones((v, v, path_length + 1)) * np.inf
    sp = np.empty_like(weights, dtype=object)
    # Init path in length 1
    for vertex_num in range(v):
        vertex = graph.get_vertex(vertex_num)
        for neighbour_num in range(v):
            neighbour = graph.get_vertex(neighbour_num)
            sp[vertex_num, neighbour_num, 1] = [vertex_num, neighbour_num]

            if vertex_num == neighbour_num:
                weights[vertex_num, neighbour_num, 1] = np.inf
            else:
                weights[vertex_num, neighbour_num, 1] = vertex.get_weight(neighbour)

    # Compute shortest path in length <= path_length
    for e in range(2, path_length + 1):
        print("calculating path in length %d" % e)
        for i in range(v):
            cur_src = graph.get_vertex(i)
            for j in range(i+1, v):
                cur_target = graph.get_vertex(j)
                for k in range(i+1, j):
                    cur_node = graph.get_vertex(k)
                    if (cur_src.get_weight(cur_node) != np.inf and
                        cur_src.id != cur_node.id and cur_target.id != cur_node.id and
                        weights[k, j, e - 1] != np.inf):
                        new_weight = cur_src.get_weight(cur_node) + weights[k, j, e - 1]
                        if new_weight < weights[i, j, e]:
                            weights[i, j, e] = new_weight
                            sp[i, j, e] = [i] + [k] + sp[k, j, e-1][1:-1] + [j]
                            cur_node.set_previous(cur_src)
                            cur_target.set_previous(cur_node)

    path = np.sort(sp[src_node.id, target_node.id, path_length])
    sp_weight = weights[src_node.id, target_node.id, path_length]
    # return weights[src_node.id, target_node.id, path_length], path
    return path, sp_weight



if __name__ == '__main__':

    g = Graph()
    for i in range(4):
        g.add_vertex(i)

    g.add_edge(0, 0, 0)
    g.add_edge(0, 1, 10)
    g.add_edge(0, 2, 3)
    g.add_edge(0, 3, 2)
    g.add_edge(0, 4, np.inf)
    g.add_edge(1, 0, np.inf)
    g.add_edge(1, 1, 0)
    g.add_edge(1, 2, 5)
    g.add_edge(1, 3, 1)
    g.add_edge(1, 4, np.inf)
    g.add_edge(2, 0, np.inf)
    g.add_edge(2, 1, np.inf)
    g.add_edge(2, 2, 0)
    g.add_edge(2, 3, 6)
    g.add_edge(2, 4, np.inf)
    g.add_edge(3, 0, np.inf)
    g.add_edge(3, 1, np.inf)
    g.add_edge(3, 2, np.inf)
    g.add_edge(3, 3, 0)
    g.add_edge(3, 4, 2)
    g.add_edge(4, 0, np.inf)
    g.add_edge(4, 1, np.inf)
    g.add_edge(4, 2, np.inf)
    g.add_edge(4, 3, np.inf)
    g.add_edge(4, 4, 0)


    w, p = shortest_path(g, g.get_vertex(0), g.get_vertex(4), 3)
    print(w)
    print(p)

    # g = Graph()
    #
    # g.add_vertex('a')
    # g.add_vertex('b')
    # g.add_vertex('c')
    # g.add_vertex('d')
    # g.add_vertex('e')
    # g.add_vertex('f')
    #
    # g.add_edge('a', 'b', 7)
    # g.add_edge('a', 'c', 9)
    # g.add_edge('a', 'f', 14)
    # g.add_edge('b', 'c', 10)
    # g.add_edge('b', 'd', 15)
    # g.add_edge('c', 'd', 11)
    # g.add_edge('c', 'f', 2)
    # g.add_edge('d', 'e', 6)
    # g.add_edge('e', 'f', 9)
    #
    # print('Graph data:')
    # for v in g:
    #     for w in v.get_connections():
    #         vid = v.get_id()
    #         wid = w.get_id()
    #         print('( %s , %s, %3d)' % (vid, wid, v.get_weight(w)))
    #
    # dijkstra(g, g.get_vertex('a'))
    #
    # target = g.get_vertex('e')
    # path = [target.get_id()]
    # shortest(target, path)
    # print('The shortest path : %s' % (path[::-1]))

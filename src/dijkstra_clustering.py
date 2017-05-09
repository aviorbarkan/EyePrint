import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dijkstra

# https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6

def calc_plane(xyz):
    # solve: z = ax + by + c (plane)
    # np.c_ slice objects to concatenation along the second axis.
    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(xyz.shape[0])]
    # C = [a, b, c]
    C, residuals, _, _ = np.linalg.lstsq(A, xyz[:, 2])  # coefficients (a, b, c)
    return C, residuals

def build_graph(x, data, debug=False):
    assert len(x) == data.shape[0], "Length of x must be equal to number of points in data"
    n_points = data.shape[0]

    # Build nodes
    graph = dijkstra.Graph()
    for i in range(n_points):
        graph.add_vertex(i)

    # Takes 3 points so that a plane could be computed at a later stage
    for i in range(n_points-5):
        for j in range(i+5, n_points):
            xyz = np.c_[x[i:j], data[i:j]]
            C, residuals = calc_plane(xyz)

            if debug:
                # evaluate it on grid
                # xx, yy = np.meshgrid(np.linspace(x[i], x[j], 10), np.linspace(data[i, 0], data[j, 0], 10))
                xx, yy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 1, 10))
                zz = C[0] * xx + C[1] * yy + C[2]

                # plot points and fitted surface
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.5)
                ax.scatter(x, data[:, 0], data[:, 1], c='r', s=50)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.axis('equal')
                ax.axis('tight')
                plt.show()

            graph.add_edge(i, j, residuals)
    return graph

def deprected_build_graph(x, data):
    """
        the old build graph version 
    """
    graph = dijkstra.Graph()

    for i in range(len(x)):
        graph.add_vertex(i)

    for i in range(len(x)-2):
        for j in range(i+2, len(x)):
            a = (data[j]-data[i])/(x[j]-x[i])
            b = data[i]-a*x[i]
            x_mat = np.tile(x[:, np.newaxis], data.shape[1])
            data_vec = a*x_mat + b
            delta = data[(i+1):j]-data_vec[(i+1):j]
            dist = np.sqrt(np.sum(delta**2))
            graph.add_edge(i, j, dist)
    return graph

def get_best_division(graph, source, target):
    """
    
    :param graph: verteices and weights between them
    :param source: the vertex from which the path calculation start
    :param target: the target vertex in which the path ends
    :return: the shortest path from the source vertex to target vertex 
             based on the weights between them
    """
    dijkstra.dijkstra(graph, source)
    path = [target.get_id()]
    dijkstra.shortest(target, path)
    print('The shortest path : %s' % (path[::-1]))
    return path[::-1]

def draw_division(x, y, path):
    """
    Display a visualization of given vertices and the path 
    between them
    
    :param x: discrete values of the vertices
    :param y: vertices
    :param path: the path between the vertices 
    """
    # Show the vertices
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y)
    ax.plot(x[path], y[path], '.-', color='red')

    plt.show()

def draw_3d_division(x, y, z, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=10, color='blue', depthshade=False)
    ax.scatter(x[path], y[path], z[path], s=20, color='red', depthshade=False)
    ax.plot(x[path], y[path], z[path], color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for n in range(len(path)-1):
        xyz = np.c_[x[path[n]:path[n+1]], y[path[n]:path[n+1]], z[path[n]:path[n+1]]]
        C, residuals = calc_plane(xyz)
        print(C)
        xx, yy = np.meshgrid(np.linspace(x[path[n]] - 0.1, x[path[n+1]] + 0.1, 10),
                             np.linspace(y[path[n]] - 0.1, y[path[n + 1]] + 0.1, 10),)
        zz = C[0] * xx + C[1] * yy + C[2]
        ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.5, color='red')

    plt.show()

if __name__ == "__main__":
    n=1
    # Create synthetic data points for the graph
    x = np.arange(0, 100)
    r1 = np.random.random(20)*0.1
    r2 = np.random.random(30)*0.2 + 0.5
    r3 = np.random.random(15)*0.7
    r4 = np.random.random(35)*0.4*np.linspace(0, 1, 35)
    r = np.concatenate((r1, r2, r3, r4))
    # g = r.copy()
    g = np.random.random((n, 100)) * 0.3 + 0.2
    b = np.random.random((n, 100)) * 0.1 + 0.3
    # data - synthetic data representing rgb values of 100 vertices
    data = np.vstack((r, g)).T

    graph = build_graph(x, data, debug=False)

    source = graph.get_vertex(0)
    target = graph.get_vertex(len(x) - 1)
    path = get_best_division(graph, source, target)
    draw_3d_division(x, data[:, 0], data[:, 1], path)
    # draw_division(x, y, path)


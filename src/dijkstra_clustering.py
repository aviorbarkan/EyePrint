import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from mpl_toolkits.mplot3d import Axes3D
import dijkstra

# https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6

def calc_plane(xyz):
    # solve: z = ax + by + c (plane)
    # np.c_ slice objects to concatenation along the second axis.
    A = np.c_[xyz[:, :-1], np.ones(xyz.shape[0])]
    # C = [a, b, c]
    C, residuals, _, _ = np.linalg.lstsq(A, xyz[:, -1])  # coefficients (a, b, c)
    return C, residuals

def build_graph(img, valid_mask, debug=False):
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    yy = yy - img.shape[0]/2
    n_points = img.shape[1]

    # Build nodes
    graph = dijkstra.Graph()
    for i in range(n_points):
        graph.add_vertex(i)

    min_points_to_include = 3  # must be >=3
    # Takes 3 points so that a plane could be computed at a later stage
    for x_start in range(n_points - min_points_to_include):
        print('calculating %d/%d' % (x_start, n_points - min_points_to_include-1))
        for x_end in range(x_start + min_points_to_include, n_points):
            cur_valid_mask = valid_mask[:, x_start:x_end]
            x_vec = xx[:, x_start:x_end][cur_valid_mask]
            y_vec = np.abs(yy[:, x_start:x_end][cur_valid_mask])
            z_vec = img[:, x_start:x_end][cur_valid_mask]
            xyz = np.c_[x_vec, y_vec, z_vec]
            C, residuals = calc_plane(xyz)

            # Debug should be at max
            # for 3d
            if debug:
                # evaluate it on grid
                zz = C[0] * xx + C[1] * yy + C[2]

                # plot points and fitted surface
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(xx, yy, zz, alpha=0.5)
                ax.scatter(x_vec, y_vec, z_vec, c='r', s=50)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.axis('equal')
                ax.axis('tight')
                ax.set_zlim([0, 255])
                plt.show()

            graph.add_edge(x_start, x_end, residuals)
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

def draw_2d_division(x, y, path):
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

def draw_3d_division(img, valid_mask, path):
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    yy = yy - img.shape[0]/2
    n_points = img.shape[1]
    xx_valid = xx[valid_mask]
    yy_valid = yy[valid_mask]
    img_valid = img[valid_mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xx_valid, yy_valid, img_valid, s=10, c=img_valid, cmap='gray', depthshade=False)
    ax.scatter(xx[:, path], yy[:, path], 0, s=10, color='red', depthshade=False)

    # ax.plot(x[path], y[path], z[path], color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim([0, 255])

    for n in range(len(path)-1):
        cur_valid_mask = valid_mask[:, path[n]:path[n+1]]
        x_vec = xx[:, path[n]:path[n+1]][cur_valid_mask]
        y_vec = yy[:, path[n]:path[n+1]][cur_valid_mask]
        z_vec = img[:, path[n]:path[n+1]][cur_valid_mask]
        xyz = np.c_[x_vec, y_vec, z_vec]
        C, residuals = calc_plane(xyz)
        print(C)

        xx_section = xx[:, path[n]:path[n+1]]
        yy_section = yy[:, path[n]:path[n+1]]
        zz = C[0] * xx_section + C[1] * yy_section + C[2]
        ax.plot_surface(xx_section, yy_section, zz, alpha=0.5, color='red')

    plt.show()

if __name__ == "__main__":
    print_path = True
    gray_input = True

    im_rgb = cv2.imread(r'../images/input.png')
    im_gray =  cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY)
    if gray_input:
        input_img = im_gray
        valid_mask = input_img < 255
    else:
        input_img = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2HSV)[:, :, 0]
        valid_mask = input_img > 0

    graph = build_graph(input_img, valid_mask, debug=False)

    source = graph.get_vertex(0)
    target = graph.get_vertex(im_rgb.shape[1] - 1)
    path = get_best_division(graph, source, target)

    if print_path:
        draw_3d_division(im_gray, valid_mask, path)

    # print the division on the image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_rgb)
    # y_path = np.ones(x[path].shape) * y_size
    # ax.bar(x[path], y_path, '.-', color='green')
    for i in path:
        rect = patches.Rectangle((i, 0), 1, im_rgb.shape[0], linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()



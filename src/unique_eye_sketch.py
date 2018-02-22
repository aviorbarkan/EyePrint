import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import data_structures
import sys
import time as time
import coreset_creator
import math
import multiprocessing
from multiprocessing import Pool

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering


###############################################################################
#                               Graph functions                               #
###############################################################################

def build_img_graph(img, valid_mask, debug=False):
    """
    Build a graph from a given image where the indices of x-axis are the nodes
    :param img: a gray-scale image
    :param valid_mask: a mask that marks the relevant elements of the image
    :param debug: flag for visualisation in debug mode
    :return: the graph that was computed
    """
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    yy = yy - img.shape[0] / 2
    n_points = img.shape[1]

    # Build nodes
    graph = data_structures.Graph()
    for i in range(n_points):
        graph.add_vertex(i)

    # Takes 3 points so that a plane could be computed at a later stage
    for x_start in range(n_points):
        graph.add_edge(x_start, x_start, 0)

        for x_end in range(x_start + 1, n_points):
            cur_valid_mask = valid_mask[:, x_start:x_end]
            x_vec = xx[:, x_start:x_end][cur_valid_mask]
            y_vec = np.abs(yy[:, x_start:x_end][cur_valid_mask])
            z_vec = img[:, x_start:x_end][cur_valid_mask]
            xyz = np.c_[x_vec, y_vec, z_vec]
            if len(xyz) > 0:
                C, residuals = calc_plane(xyz)

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

            # one directional graph - the weights in the opposite direction are inf
            else:
                residuals = np.inf

            if x_end == x_start+1:
                graph.add_edge(x_start, x_end,  np.inf)
                graph.add_edge(x_end, x_start, np.inf)
            else:
                graph.add_edge(x_start, x_end, residuals)
                graph.add_edge(x_end, x_start, np.inf)

    return graph


def build_degrees_graph(eye_img, center, radius, min_degree, degree_graph_fn, img_width=None,
                        start_at_degree=0, stop_at_degree=360, load_graph_fn=None):
    """
    Build a graph from a given image, where the angles are the nodes
    :param eye_img: a gray-scale image
    :param center: the center coordinate of the eye
    :param radius: eye radius length
    :param min_degree: for partial calculation
    :param degree_graph_fn: for saving the graph
    :param img_width: optional - to work on a costume sized image
    :param start_at_degree: for partial calculation - start computing from a certain angle
    :param stop_at_degree: for partial calculation - stop computing at a certain angle
    :param load_graph_fn: for start computing from a partially computed graph
    :return: the graph that was computed
    """

    n_points = int(360/min_degree) + 1  # 360 degrees
    starting_point = int(start_at_degree/min_degree)
    ending_point = int(stop_at_degree/min_degree) + 1

    # Build nodes - one for each degree
    if load_graph_fn is None:
        # Build all graph nodes
        graph = data_structures.Graph()
        for i in range(n_points):
            graph.add_vertex(i)
    else:
        # Continue calculating a given graph
        print("Loading from: %s" % load_graph_fn)
        with open(load_graph_fn, 'rb') as f:
            graph = pickle.load(f)
        backup_fn = load_graph_fn[:-4] + "_backup.pkl"
        print("Saving to: %s" % backup_fn)
        with open(backup_fn, 'wb') as f:
            pickle.dump(graph, f)

    results = [None] * n_points
    pool = Pool(processes=multiprocessing.cpu_count() - 1)
    # This loop can also be converted to a map-reduce operation. This loop is what makes the algorithm work forever.
    for index in range(starting_point, ending_point, 1):
        results[index] = pool.apply_async(calculate_graph_from_degree_to_degree_parallel,
                                          args=(index, n_points, min_degree, img_width, graph, center, radius, eye_img))
    pool.close()
    pool.join()
    count = 0
    try:
        for result in results:
            count += 1
            graph.concat_graph(result.get())  # Joins the resulted graphs.
    except:
        print "The graph concatenation received an exception. Ony concatenated " + str(count) + " out of " + \
                str(n_points) + ". You are probably using not enough points in the image. Try larger dataset."

    print("Saving to: %s" % degree_graph_fn)
    with open(degree_graph_fn, 'wb') as f:
        pickle.dump(graph, f)

    return graph


def calculate_graph_from_degree_to_degree_parallel(x_start, n_points, min_degree, img_width, graph, center, radius, eye_img):
    graph.add_edge(x_start, x_start, 0)

    for x_end in range(x_start + 1, n_points):
        # get the sector between x_start and x_end
        angles = np.array([np.deg2rad(x_start * min_degree), np.deg2rad(x_end * min_degree)])
        print('calculating graph from degree %d to degree %d' % (x_start * min_degree, x_end * min_degree))
        sector = img_to_sectors(eye_img, center, radius, angles_list=angles, plot_debug=False)
        slice_valid_mask = sector[0]['mask_rotated']
        slice_img = sector[0]['img_sector_rotated']

        valid_points = np.argwhere(slice_valid_mask == True)
        x_min_valid = np.min(valid_points[:, 1])
        x_max_valid = np.max(valid_points[:, 1])
        y_min_valid = np.min(valid_points[:, 0])
        y_max_valid = np.max(valid_points[:, 0])
        cropped_slice_img = slice_img[y_min_valid:y_max_valid + 1, x_min_valid:x_max_valid + 1]
        cropped_slice_valid_mask = slice_valid_mask[y_min_valid:y_max_valid + 1, x_min_valid:x_max_valid + 1]
        slice_path, slice_error = get_slice_partition(cropped_slice_img, cropped_slice_valid_mask, display=False,
                                                      width=img_width)
        graph.add_edge(x_start, x_end, slice_error)
        graph.add_edge(x_end, x_start, np.inf)

    return graph


###############################################################################
#                                Aux functions                                #
###############################################################################
def shortest_path(graph, src_node, target_node, path_length, given_weights=None, given_sp=None):
    """
    Given a directed graph and two vertices 'u' and 'v' in it,
    find shortest path from 'u' to 'v' with exactly k edges on the path.
    :param graph: data set
    :param src_node: where the path should start
    :param target_node: where the path should end
    :param path_length: the requiered path length
    :param given_weights: pre-calculated weights in a given depth
    :param given_sp: pre-calculated paths in a given depth
    :return: chosen path, it's weight, all weight matrix, all possible paths matrix
    """
    v = graph.num_vertices
    if path_length == 0:
        if src_node.id == target_node.id:
            return [src_node.id, target_node.id]
        else:
            return np.inf

    weights = np.ones((v, v, path_length + 1)) * np.inf
    sp = np.empty_like(weights, dtype=object)

    # check if there are pre-calculated weights or paths, if so - start calculation from that point
    if given_weights is None or given_sp is None:
        e_init = 2
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
    # start the calculation from the first depth
    else:
        e_init = given_weights.shape[2]
        if given_weights.shape[2] <= path_length:
            weights[:, :, :given_weights.shape[2]] = given_weights
            sp[:, :, :given_sp.shape[2]] = given_sp
        else:
            weights = given_weights
            sp = given_sp

    # Compute shortest path in length <= path_length
    for e in range(e_init, path_length + 1):
        # print("calculating path in length %d" % e)
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

    path = sp[src_node.id, target_node.id, path_length]
    p_length = path_length
    while path is None:
        p_length = p_length - 1
        path = sp[src_node.id, target_node.id, p_length]

    path = np.sort(path)
    sp_weight = weights[src_node.id, target_node.id, p_length]

    # return chosen path, it's weight, all weight matrix, all possible paths matrix
    return path, sp_weight, weights, sp


def calc_plane(xyz):
    """
    calculate plane for given data
    :param xyz: data for computing plane
    :return: coefficients (C), residuals
    """

    # solve: z = ax + by + c (plane)
    # np.c_ slice objects to concatenation along the second axis.
    A = np.c_[xyz[:, :-1], np.ones(xyz.shape[0])]
    # C = [a, b, c]
    C, residuals, _, _ = np.linalg.lstsq(A, xyz[:, -1])  # coefficients (a, b, c)
    if len(residuals) == 0:
        residuals = 0
    return C, residuals


def find_line(first_point, second_point):
    """
    given two points, find the line between them
    :param first_point:
    :param second_point:
    :return: line equation coefficients (a, b)
    """

    # find line equation y=ax+b
    delta_x = np.float32(second_point[0]) - np.float32(first_point[0])
    delta_y = np.float32(second_point[1]) - np.float32(first_point[1])
    a = delta_y / delta_x
    b = first_point[1] - a * first_point[0]

    return a, b


def dist_from_line(a, b, point):
    """
    given line equation coefficients and a point, computes the distance of the point
    from the line
    :param a: a coefficient from y=ax+b
    :param b: b coefficient from y=ax+b
    :param point: x,y point
    :return: the distance of a point from a line with given a and b (y=ax+b)
    """

    numerator = a * point[0] - point[1] + b
    denominator = np.sqrt(a ** 2 + 1)
    dist = np.abs(numerator / denominator)

    return dist


def find_optimal_divisions_num(graph, source, target, display=False):
    """
    given a graph, finds the optimal number of divisions of the graph, according
    to it's sorce and target nodes, using binary search.
    :param graph:
    :param source: source node in the graph
    :param target: target node in the graph
    :param display: flag for visualisations
    :return: optimal division number, optimal path from source node to target node,
             the error of the optimal division
    """

    # find f(1) and f(n)
    _, f_one, init_weights, init_sp = shortest_path(graph, source, target, 1)
    _, f_n, all_weights, all_sp = shortest_path(graph, source, target, target.id, init_weights, init_sp)
    f_one = f_one - f_n
    f_n = f_n - f_n
    print(f_one)
    print(f_n)

    # find the line that go through (1,f(1)) and (n,f(n))
    a, b = find_line((1, f_one), (target.id, f_n))

    # Do binary search for the optimal number of divisions (k)
    low = 1
    high = target.id

    optimal_division = 0
    optimal_division_dist = 0
    optimal_division_error = 0
    optimal_path = []
    points_list = []
    points_list.append((1, f_one))
    while low < high:
        k = int((float(high) + float(low)) / 2.0)
        k_neighbour = k + 1

        # Get error for k divisions and k+1 divisions
        path_k, f_k, _, _ = shortest_path(graph, source, target, k-source.id, all_weights, all_sp)
        path_k_neighbour, f_k_neighbour, _, _ = shortest_path(graph, source, target, k_neighbour-source.id,
                                                              all_weights, all_sp)

        dist_k = dist_from_line(a, b, (k, f_k))
        dist_k_neighbour = dist_from_line(a, b, (k_neighbour, f_k_neighbour))

        # binary search should go left
        if dist_k > dist_k_neighbour:
            if optimal_division_dist > dist_k:
                if optimal_division > k:
                    # local maximum, ignore and go right
                    low = k_neighbour
                else:
                    high = k
            else:
                optimal_division = k
                optimal_division_dist = dist_k
                optimal_division_error = f_k
                optimal_path = path_k
                high = k

        # binary search should go right
        else:
            if optimal_division_dist > dist_k:
                if optimal_division < k:
                    # local maximum, ignore and go left
                    high = k
                else:
                    low = k_neighbour
            else:
                optimal_division = k
                optimal_division_dist = dist_k
                optimal_division_error = f_k
                optimal_path = path_k
                low = k_neighbour

        points_list.append((k, f_k))

    points_list.append((target.id, f_n))
    points_list.sort(key=lambda tup: tup[0])
    print("optimal division: k=%d" % optimal_division)
    if display:
        draw_lines(points_list, connect_first_and_last=True)
    return optimal_division, optimal_path, optimal_division_error


def img_to_num_sectors(img, center, radius, num_sectors=10, plot_debug=False):
    """
    :param img: 2D grayscale image, uint8.
    :param center: Tuple (x,y)
    :param radius: radius in pixels
    :param num_sectors: Number of sectors
    :param plot_debug: plots all sectors
    :return: list of dictonary items. Each item contains:
        - 'img_sector_rotated': Image of the valid sector, after rotation
        - 'mask_rotated': valid mask, after rotation
        - 'img_sector': Image of the valid sector, before rotation
        - 'mask': valid mask, before rotation
        - 'ang1': Start angle, in radians
        - 'ang2': End angle, in radians
        - 'ang_center': Center angle, in radians
    """
    angles = np.linspace(0, np.pi * 2, num_sectors + 1)
    return img_to_sectors(img, center, radius, angles_list=angles, plot_debug=plot_debug)


def img_to_sectors(img, center, radius, angles_list, plot_debug=False):
    """
    :param img: 2D grayscale image, uint8.
    :param center: Tuple (x,y)
    :param radius: radius in pixels
    :param angles_list: List of angles
    :param plot_debug: plots all sectors
    :return: list of dictonary items. Each item contains:
        - 'img_sector_rotated': Image of the valid sector, after rotation
        - 'mask_rotated': valid mask, after rotation
        - 'img_sector': Image of the valid sector, before rotation
        - 'mask': valid mask, before rotation
        - 'ang1': Start angle, in radians
        - 'ang2': End angle, in radians
        - 'ang_center': Center angle, in radians
    """
    # Creating grid of polar coordinates
    uu, vv = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    xx = uu - center[0]
    yy = vv - center[1]
    rr = np.sqrt(xx ** 2 + yy ** 2)
    theta = -1 * np.arctan2(yy, xx)
    theta[theta < 0] = theta[theta < 0] + 2 * np.pi
    mask_radius = rr < radius

    # Creating sectors
    result = []
    for ind in range(1, len(angles_list)):
        ang1 = angles_list[ind - 1]
        ang2 = angles_list[ind]
        mask_angle = np.logical_and(theta >= ang1, theta < ang2)
        mask = np.logical_and(mask_radius, mask_angle)
        img_sector = img * mask
        ang = (ang2 + ang1) / 2.0
        M = cv2.getRotationMatrix2D(center, -ang * 180 / np.pi, 1.0)
        img_sector_rotated = cv2.warpAffine(img_sector, M, (img.shape[1], img.shape[0]))
        mask_rotated = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (img.shape[1], img.shape[0]))
        mask_rotated = mask_rotated > 127
        result.append({'img_sector_rotated': img_sector_rotated,
                       'mask_rotated': mask_rotated,
                       'img_sector': img_sector,
                       'mask': mask,
                       'ang1': ang1,
                       'ang2': ang2,
                       'ang_center': ang})
    if plot_debug:
        for sector in result:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(sector['img_sector'], cmap='gray')
            ax.set_title("Sector %d | Angle between (%.2f, %.2f)" % (ind, ang1, ang2))
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(sector['img_sector_rotated'], cmap='gray')
            ax.set_title("Rotated Sector")
        plt.show()

    return result


def get_slice_partition(img, img_mask, display=False, width=None):
    """
    Given a slice image and a mask for the valid parts, returns the optimal
    partition of the slice by minimizing the errors from optimal plane, using
    linear regression and finding the "elbow"/"knee" point in the error graph
    between (1,f(1)) and (n, f(n)) with binary search.
    :param img: slice image
    :param img_mask: mask of valid parts of the image
    :param display: show slice results
    :param width: if given - work on a resized image in the given weights
    :return: the path (optimal division) of the original slice and the division error
    """

    im_gray = img.copy()
    orig_valid_mask = img_mask

    # Use original image size
    if width is None:
        valid_mask = img_mask
        input_img = im_gray

    # Resize the image
    else:
        r = float(width) / im_gray.shape[1]
        dim = (width, max(1, int(im_gray.shape[0] * r)))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(im_gray, dim, interpolation=cv2.INTER_AREA)
        input_img = resized

        mask_resized = cv2.resize(img_mask.astype(np.uint8) * 255, dim, interpolation=cv2.INTER_AREA)
        mask_resized = mask_resized > 127
        valid_mask = mask_resized

    graph = build_img_graph(input_img, valid_mask, debug=False)

    source = graph.get_vertex(0)
    target = graph.get_vertex(input_img.shape[1] - 1)
    # new code for shortest path with exactly k edges
    division_num, path, division_error = find_optimal_divisions_num(graph, source, target, display)
    print("slice path: ")
    print(path)

    # Original image size was used
    if width is None:
        orig_path = path

    # The image was resized, adjust path to the full size image path
    else:
        orig_path = np.int32(path / r)
        orig_path[-1] = (img.shape[1] - 1)

    if display:
        # print the division on the image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        # y_path = np.ones(x[path].shape) * y_size
        # ax.bar(x[path], y_path, '.-', color='green')
        ax.vlines(orig_path, ymin=0, ymax=img.shape[0], color='red')

        plt.show()

        # 3D display
        # draw_3d_division(im_gray, orig_valid_mask, orig_path)

    return orig_path, division_error

###############################################################################
#                               Visualizations                                #
###############################################################################


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
    """
    Display a visualization of given img and its partition according to given path
    :param img: input image
    :param valid_mask: mask of the relevant elements in the image
    :param path: the img partition
    :return:
    """

    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    yy = yy - img.shape[0] / 2
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

    for n in range(len(path) - 1):
        cur_valid_mask = valid_mask[:, path[n]:path[n + 1]]
        x_vec = xx[:, path[n]:path[n + 1]][cur_valid_mask]
        y_vec = yy[:, path[n]:path[n + 1]][cur_valid_mask]
        z_vec = img[:, path[n]:path[n + 1]][cur_valid_mask]
        xyz = np.c_[x_vec, y_vec, z_vec]
        C, residuals = calc_plane(xyz)
        print(C)

        xx_section = xx[:, path[n]:path[n + 1]]
        yy_section = yy[:, path[n]:path[n + 1]]
        zz = C[0] * xx_section + C[1] * yy_section + C[2]
        ax.plot_surface(xx_section, yy_section, zz, alpha=0.5, color='red')

    plt.show()


def plot_sectors(sectors, center, radius, img_rgb, output_fn):
    """
    display the partition of sectors
    """

    for sector in sectors:
        mask_rotated = sector['mask_rotated']
        path = sector['path']
        ang_center = sector['ang_center']

        # Compute path xy location
        y_min = np.argmax(mask_rotated[:, path], axis=0)
        y_max = len(mask_rotated) - 1 - np.argmax(mask_rotated[:, path][::-1, :], axis=0)

        xy1_hom = np.vstack((path, y_min, np.ones_like(path)))
        xy2_hom = np.vstack((path, y_max, np.ones_like(path)))
        R = cv2.getRotationMatrix2D(center, np.rad2deg(ang_center), 1.0)
        xy1_rotated = R.dot(xy1_hom).T
        xy2_rotated = R.dot(xy2_hom).T

        sector['xy1'] = xy1_hom[:, :2]
        sector['xy2'] = xy2_hom[:, :2]
        sector['xy1_rotated'] = xy1_rotated
        sector['xy2_rotated'] = xy2_rotated

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img_rgb)
    for sector_ind, sector in enumerate(sectors):
        for i in range(len(sector['xy1_rotated']) - 1):
            ax.plot([sector['xy1_rotated'][i, 0], sector['xy2_rotated'][i, 0]],
                    [sector['xy1_rotated'][i, 1], sector['xy2_rotated'][i, 1]],
                    color='red', linewidth=0.5)
        ax.plot([center[0], np.cos(-sector['ang1']) * radius + center[0]],
                [center[1], np.sin(-sector['ang1']) * radius + center[1]],
                color='red', linewidth=0.5)
        ax.plot([center[0], np.cos(-sector['ang2']) * radius + center[0]],
                [center[1], np.sin(-sector['ang2']) * radius + center[1]],
                color='red', linewidth=0.5)
        ax.plot([np.cos(-sector['ang1']) * radius + center[0], np.cos(-sector['ang2']) * radius + center[0]],
                [np.sin(-sector['ang1']) * radius + center[1], np.sin(-sector['ang2']) * radius + center[1]],
                color='red', linewidth=0.5)
        ax.text(np.cos(-sector['ang_center']) * radius + center[0],
                np.sin(-sector['ang_center']) * radius + center[1],
                '%d' % sector_ind,
                verticalalignment='top', horizontalalignment='center',
                color='red', fontsize=10)
    ax.set_ylim([img_rgb.shape[0], 0])
    fig.tight_layout()
    fig.savefig(output_fn)
    plt.show()


def draw_lines(points_list, connect_first_and_last=False):
    """
    given a list of points, draw a line between each two
    :param points_list: list of x,y points
    :param connect_first_and_last: flag. if true - draw a line between the
           first point and last point.
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(points_list) - 1):
        p1 = points_list[i]
        p2 = points_list[i + 1]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

    if connect_first_and_last:
        plt.plot([points_list[0][0], points_list[-1][0]], [points_list[0][1], points_list[-1][1]])

    ax.set_xlabel('Number of divisions')
    ax.set_ylabel('Error')
    plt.show()


def unique_eye_sketch(rgb_eye_img, coreset_size, min_degree, with_coreset, agglomerative=True, width=None):
    """
    :param rgb_eye_img: and rgb eye image
    :param center: the center coordinates of the eye (x,y)
    :param radius: eye radius
    :param agglomerative: flag for the use of agglomerative clustering
    :return: the eye image with unique eye print segmentation
    """

    print "Starting time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    sys.setrecursionlimit(2500)
    output_img_fn = 'unique_eye_sketch.png'

    output_img = rgb_eye_img.copy()
    orig_eye_img = cv2.cvtColor(rgb_eye_img, cv2.COLOR_RGB2GRAY)

    # Paths for saving/loading the computed data
    sectors_fn = "../data/sectors.pkl"
    graph_fn = "../data/degrees_graph.pkl"
    path_fn = "../data/degrees_path.pkl"

    orig_center = (int(orig_eye_img.shape[1] / 2.0), int(orig_eye_img.shape[0] / 2.0))
    orig_radius = int(orig_eye_img.shape[0] * 0.4);
    if with_coreset is True:
        coreset_eye_img = coreset_creator.create(orig_eye_img, orig_eye_img.shape[0], orig_eye_img.shape[1], coreset_size)  # Create coreset
        new_height = int(math.sqrt((float(orig_eye_img.shape[0]) / orig_eye_img.shape[1]) * coreset_size))  # compressed image new height
        new_width = int(coreset_size / new_height)  # compressed image new width
        new_center = (int(new_width / 2.0), int(new_height / 2.0))  # Compressed image new center of eye image
        new_radius = int(new_height * 0.4);
        # Had some trouble estimating the compressed image new eye radius. If you have a better idea - go ahead.
        degree_graph = build_degrees_graph(coreset_eye_img, new_center, new_radius, min_degree=min_degree,
                                           degree_graph_fn=graph_fn, img_width=width)
    else:
        degree_graph = build_degrees_graph(orig_eye_img, orig_center, orig_radius, min_degree=min_degree,
                                           degree_graph_fn=graph_fn, img_width=width)

    source = degree_graph.get_vertex(0)
    target = degree_graph.get_vertex(int(360 / min_degree))
    print "Ending time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # Find the optimal division for the whole eye
    opt_division, opt_path_divided, opt_division_error = find_optimal_divisions_num(degree_graph, source,
                                                                                    target, display=True)
    optimal_path = opt_path_divided * min_degree

    print("Saving optimal degrees path to: %s" % path_fn)
    with open(path_fn, 'wb') as f:
        pickle.dump(optimal_path, f)

    print("optimal degree path: ")
    print(optimal_path)

    angles_path = np.deg2rad(optimal_path)
    sectors = img_to_sectors(orig_eye_img, orig_center, orig_radius, angles_list=angles_path, plot_debug=False)

    # The agglomerative algorithm needs work to make it work. Currently it throws an exception. Use the alternative.
    # use agglomerative clustering for the segmentation of each slice
    if agglomerative:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(output_img, cmap=plt.cm.gray)
        for sector_ind, sector in enumerate(sectors):
            slice_valid_mask = sector['mask']
            slice_img_rotated = sector['img_sector_rotated']
            X = np.reshape(slice_img_rotated, (-1, 1))

            connectivity = grid_to_graph(*slice_img_rotated.shape)

            print("Compute structured hierarchical clustering %d/%d" % (sector_ind, len(sectors)))
            n_clusters = 8  # number of regions
            ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                                           connectivity=connectivity)
            ward.fit(X)
            clusters_labels = np.reshape(ward.labels_, slice_img_rotated.shape)
            max_label = np.max(clusters_labels)
            clusters_labels = np.round(clusters_labels * 256 / max_label)
            # rotate labels to original slice position
            ang = (sector['ang2'] + sector['ang1']) / 2.0
            M = cv2.getRotationMatrix2D(orig_center, ang * 180 / np.pi, 1.0)
            labels_rotated = cv2.warpAffine(clusters_labels, M, (clusters_labels.shape[1],
                                                                 clusters_labels.shape[0]))
            labels_rotated = np.round(labels_rotated * max_label / 256)
            labels_rotated[~slice_valid_mask] = -1
            sector['labels'] = labels_rotated

            valid_labels = np.unique(labels_rotated)
            valid_labels = valid_labels[valid_labels >= 0]
            for l in valid_labels:
                plt.contour(labels_rotated == l, contours=1, colors='r')

            ax.text(np.cos(-sector['ang_center']) * orig_radius + orig_center[0],
                    np.sin(-sector['ang_center']) * orig_radius + orig_center[1],
                    '%d' % sector_ind,
                    verticalalignment='top', horizontalalignment='center',
                    color='blue', fontsize=10)

        plt.xticks(())
        plt.yticks(())
        fig.tight_layout()
        fig.savefig(output_img_fn)
        plt.show()

    else:
        for sector in sectors:
            slice_valid_mask = sector['mask_rotated']
            slice_img = sector['img_sector_rotated']

            valid_points = np.argwhere(slice_valid_mask == True)
            x_min_valid = np.min(valid_points[:, 1])
            x_max_valid = np.max(valid_points[:, 1])
            y_min_valid = np.min(valid_points[:, 0])
            y_max_valid = np.max(valid_points[:, 0])
            cropped_slice_img = slice_img[y_min_valid:y_max_valid, x_min_valid:x_max_valid]
            cropped_slice_valid_mask = slice_valid_mask[y_min_valid:y_max_valid, x_min_valid:x_max_valid]

            cropped_slice_path, cropped_slice_error = get_slice_partition(cropped_slice_img,
                                                                          cropped_slice_valid_mask,
                                                                          display=False, width=width)

            slice_path = cropped_slice_path + x_min_valid
            sector['path'] = slice_path
        # Plot the final sectors of the eye
        print('Optimal division results:')
        plot_sectors(sectors, center=orig_center, radius=orig_radius, img_rgb=output_img, output_fn=output_img_fn)

    print("Saving to: %s" % sectors_fn)
    with open(sectors_fn, 'wb') as f:
        pickle.dump(sectors, f)

    final_img = cv2.imread(output_img_fn)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    return final_img


if __name__ == "__main__":
    width = None  # When not None, used to compress the image. doesn't work well with an already compressed image
    min_degree = 5  # Divides each section to this amount of degrees. Increases code complexity when lower.

    # Load eye image
    input_img = cv2.imread('..\images\slice-21.jpg')
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # Don't run with coreset size less than 5000. The algorithm crashes when there's not enough points to run on.
    # The center of the picture and the new radius are calculated within the code now according to the requested
    # coreset size.
    eye_img = unique_eye_sketch(rgb_eye_img=img_rgb, coreset_size=12000, min_degree=min_degree, with_coreset=True, agglomerative=False, width=width)


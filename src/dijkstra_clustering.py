import os
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pickle
from mpl_toolkits.mplot3d import Axes3D
import dijkstra


# https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6

def calc_plane(xyz):
    # solve: z = ax + by + c (plane)
    # np.c_ slice objects to concatenation along the second axis.
    A = np.c_[xyz[:, :-1], np.ones(xyz.shape[0])]
    # C = [a, b, c]
    C, residuals, _, _ = np.linalg.lstsq(A, xyz[:, -1])  # coefficients (a, b, c)
    if len(residuals) == 0:
        residuals = 0
    return C, residuals


def build_img_graph(img, valid_mask, debug=False):
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    yy = yy - img.shape[0] / 2
    n_points = img.shape[1]

    # Build nodes
    graph = dijkstra.Graph()
    for i in range(n_points):
        graph.add_vertex(i)

    # Takes 3 points so that a plane could be computed at a later stage
    for x_start in range(n_points):
        #print('calculating %d/%d' % (x_start, n_points))

        graph.add_edge(x_start, x_start, 0)

        for x_end in range(x_start + 1, n_points):
            cur_valid_mask = valid_mask[:, x_start:x_end]
            x_vec = xx[:, x_start:x_end][cur_valid_mask]
            y_vec = np.abs(yy[:, x_start:x_end][cur_valid_mask])
            z_vec = img[:, x_start:x_end][cur_valid_mask]
            xyz = np.c_[x_vec, y_vec, z_vec]
            if len(xyz) > 0:
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

            else:
                residuals = np.inf

            graph.add_edge(x_start, x_end, residuals)
            graph.add_edge(x_end, x_start, np.inf)

    return graph


def build_degrees_graph(img, center, radius, min_degree=1, debug=False):
    degree_graph_fn = "degree_graph_v4.pkl"
    n_points = int(360/min_degree) + 1  # 360 degrees

    # Build nodes - one for each degree
    graph = dijkstra.Graph()
    for i in range(n_points):
        graph.add_vertex(i)

    # Takes 3 points so that a plane could be computed at a later stage
    for x_start in range(n_points):
        graph.add_edge(x_start, x_start, 0)

        for x_end in range(x_start + 1, n_points):
            # get the sector between x_start and x_end
            angles = np.array([np.deg2rad(x_start), np.deg2rad(x_end*min_degree)])
            print('calculating graph from degree %d to degree %d' % (x_start*min_degree, x_end*min_degree))
            sector = img_to_sectors(eye_img, center, radius, angles_list=angles, plot_debug=False)
            slice_valid_mask = sector[0]['mask_rotated']
            slice_img = sector[0]['img_sector_rotated']

            valid_points = np.argwhere(slice_valid_mask == True)
            x_min_valid = np.min(valid_points[:, 1])
            x_max_valid = np.max(valid_points[:, 1])
            y_min_valid = np.min(valid_points[:, 0])
            y_max_valid = np.max(valid_points[:, 0])
            cropped_slice_img = slice_img[y_min_valid:y_max_valid, x_min_valid:x_max_valid]
            cropped_slice_valid_mask = slice_valid_mask[y_min_valid:y_max_valid, x_min_valid:x_max_valid]

            slice_path, slice_error = get_slice_partition(cropped_slice_img, cropped_slice_valid_mask, display=False,
                                                          original_size=False, width=64)
            graph.add_edge(x_start, x_end, slice_error)
            graph.add_edge(x_end, x_start, np.inf)
        print("Saving to: %s" % degree_graph_fn)
        with open(degree_graph_fn, 'wb') as f:
            pickle.dump(graph, f)
    return graph

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


def find_line(first_point, second_point):
    # find line equation y=ax+b
    delta_x = np.float32(second_point[0]) - np.float32(first_point[0])
    delta_y = np.float32(second_point[1]) - np.float32(first_point[1])
    a = delta_y / delta_x
    b = first_point[1] - a * first_point[0]

    return a, b


def draw_lines(points_list, connect_first_and_last=False):
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


def dist_from_line(a, b, point):
    # returns the distance of a point from a line with given a and b (y=ax+b)
    numerator = a * point[0] - point[1] + b
    denominator = np.sqrt(a ** 2 + 1)
    d = np.abs(numerator / denominator)

    return d


def find_optimal_divisions_num(graph, source, target, display=False):
    # find f(1) and f(n)
    _, f_one, _, _ = dijkstra.shortest_path(graph, source, target, 1)
    print(f_one)
    _, f_n, all_weights, all_sp = dijkstra.shortest_path(graph, source, target, target.id)
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
        #print("low = %d, k = %d, high = %d" % (low, k, high))
        # Get error for k divisions andty k+1 divisions
        path_k, f_k, _, _ = dijkstra.shortest_path(graph, source, target, k-source.id, all_weights, all_sp)
        path_k_neighbour, f_k_neighbour, _, _ = dijkstra.shortest_path(graph, source, target, k_neighbour-source.id,
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
    print("optimal division: k=%d" % k)
    # if display:
    #     draw_lines(points_list, connect_first_and_last=True)
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
        mask_angle = np.logical_and(theta > ang1, theta < ang2)
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


def get_slice_partition(img, img_mask, display=False, original_size=True, width=200):
    im_gray = img.copy()
    orig_valid_mask = img_mask

    if original_size:
        valid_mask = img_mask
        input_img = im_gray

    else:
        r = float(width) / im_gray.shape[1]
        dim = (width, int(im_gray.shape[0] * r))
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
    if original_size:
        orig_path = path
    else:
        orig_path = np.int32(path / r)
        orig_path[-1] = (img.shape[1] - 1)
        # dijkstra old code
        # path = get_best_division(graph, source, target)

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

def plot_sectors(sectors, center, radius, img_rgb):
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
    fig.savefig("eye_sectors2.png")
    plt.show()

if __name__ == "__main__":
    calculate = True
    # Load eye image
    input_img = cv2.imread(r'../images/src.jpg')
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    sectors_fn = "sectors_v4.pkl"
    eye_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    if calculate:
        min_degree = 36

        degree_graph = build_degrees_graph(eye_img, center=(501, 273), radius=285, min_degree=min_degree, debug=False)

        source = degree_graph.get_vertex(0)
        target = degree_graph.get_vertex(int(360/min_degree))
        optimal_division, optimal_path_divided, optimal_division_error = find_optimal_divisions_num(degree_graph,
                                                                                                    source, target)
        optimal_path = optimal_path_divided * min_degree
    else:
        # optimal_division = 67
        # optimal_path = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69,
        #                 72, 75, 78, 81, 84, 87, 90, 96, 99, 102, 105, 108, 114, 117, 126, 135, 138, 141, 144, 147,
        #                 150, 153, 159, 162, 171, 177, 291, 294, 297, 300, 303, 306, 309, 312, 315, 321, 327, 330,
        #                 333, 336, 345, 354, 357, 360]

        num_sectors = 22
        # optimal_path = np.linspace(0, 360, num_sectors + 1)
        optimal_path = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
                        192, 360]

    print("optimal degree path: ")
    print(optimal_path)
    angles_path = np.deg2rad(optimal_path)
    # sectors = img_to_num_sectors(eye_img, center=(501, 273), radius=285, num_sectors=10, plot_debug=False)

    # angles = np.deg2rad(np.arange(0, 360))
    if os.path.isfile(sectors_fn):
        print("Loading from: %s" % sectors_fn)
        with open(sectors_fn, 'rb') as f:
            sectors = pickle.load(f)
    else:
        sectors = img_to_sectors(eye_img, center=(501, 273), radius=285, angles_list=angles_path, plot_debug=False)

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

            cropped_slice_path, cropped_slice_error = get_slice_partition(cropped_slice_img, cropped_slice_valid_mask,
                                                                          display=False, original_size=False, width=64)

            slice_path = cropped_slice_path + x_min_valid
            sector['path'] = slice_path
        print("Saving to: %s" % sectors_fn)
        with open(sectors_fn, 'wb') as f:
            pickle.dump(sectors, f)
    print('')
    plot_sectors(sectors, center=(501, 273), radius=285, img_rgb=img_rgb)


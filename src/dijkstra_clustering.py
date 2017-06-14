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
    if len(residuals) == 0:
        residuals = 0
    return C, residuals

def build_graph(img, valid_mask, debug=False):
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    yy = yy - img.shape[0]/2
    n_points = img.shape[1]

    # Build nodes
    graph = dijkstra.Graph()
    for i in range(n_points):
        graph.add_vertex(i)

    # Takes 3 points so that a plane could be computed at a later stage
    for x_start in range(n_points):
        print('calculating %d/%d' % (x_start, n_points))

        graph.add_edge(x_start, x_start, 0)

        for x_end in range(x_start + 1, n_points):
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
            graph.add_edge(x_end, x_start, np.inf)

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


def find_line(first_point, second_point):
    # find line equation y=ax+b
    delta_x = np.float32(second_point[0])-np.float32(first_point[0])
    delta_y = np.float32(second_point[1])-np.float32(first_point[1])
    a = delta_y/delta_x
    b = first_point[1]-a*first_point[0]

    return a, b


def draw_lines(points_list, connect_first_and_last=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(points_list)-1):
        p1 = points_list[i]
        p2 = points_list[i+1]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

    if connect_first_and_last:
        plt.plot([points_list[0][0], points_list[-1][0]], [points_list[0][1], points_list[-1][1]])

    ax.set_xlabel('Number of divisions')
    ax.set_ylabel('Error')
    plt.show()

def dist_from_line(a,b, point):
    # returns the distance of a point from a line with given a and b (y=ax+b)
    numerator = a * point[0] - point[1] + b
    denominator = np.sqrt(a**2 + 1)
    d = np.abs(numerator/denominator)

    return d


def find_optimal_divisions_num(graph, source, target):
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
        k = int((float(high) + float(low))/2.0)
        k_neighbour = k+1
        print("low = %d, k = %d, high = %d" % (low, k, high))
        # Get error for k divisions and k+1 divisions
        path_k, f_k, _, _ = dijkstra.shortest_path(graph, source, target, k,  all_weights, all_sp)
        path_k_neighbour, f_k_neighbour, _, _ = dijkstra.shortest_path(graph, source, target, k_neighbour, all_weights, all_sp)

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
                low = k

        points_list.append((k, f_k))

    points_list.append((target.id, f_n))
    points_list.sort(key=lambda tup: tup[0])
    print("optimal division: k=%d" % k)
    draw_lines(points_list, connect_first_and_last=True)
    return optimal_division, optimal_path, optimal_division_error

def img_to_sectors(img, center, radius, num_sectors=10, plot_debug=False):
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
    # Creating grid of polar coordinates
    uu, vv = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    xx = uu - center[0]
    yy = vv - center[1]
    rr = np.sqrt(xx**2 + yy**2)
    theta = -1*np.arctan2(yy, xx)
    theta[theta < 0] = theta[theta < 0] + 2*np.pi
    mask_radius = rr < radius

    # Creating sectors
    angles = np.linspace(0, np.pi * 2, num_sectors + 1)
    result = []
    for ind in range(1, len(angles)):
        ang1 = angles[ind-1]
        ang2 = angles[ind]
        mask_angle = np.logical_and(theta > ang1, theta < ang2)
        mask = np.logical_and(mask_radius, mask_angle)
        img_sector = img*mask
        ang = (ang2 + ang1) / 2.0
        M = cv2.getRotationMatrix2D(center, -ang*180/np.pi, 1.0)
        img_sector_rotated = cv2.warpAffine(img_sector, M, (img.shape[1], img.shape[0]))
        mask_rotated = cv2.warpAffine(mask.astype(np.uint8)*255, M, (img.shape[1], img.shape[0]))
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

    # slices = []
    # optimal_slices_division = 0
    # optimal_division_error = 0
    #
    # # num_of_slices
    # img_debug = img.copy()
    # cv2.circle(img_debug, center, radius, [100, 255, 255], 3)
    # cv2.circle(img_debug, center, 1, [0, 255, 0], 2)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(img_debug)
    #
    # division_error = 0
    # for slice_img in slices:
    #     slice_path, slice_error = get_slice_partition(slice_img)
    #     division_error = division_error + slice_error


def get_slice_partition(im_rgb, display=False):
    original_size = False
    im_gray = im_rgb.copy()
    im_gray = cv2.cvtColor(im_gray, cv2.COLOR_BGR2GRAY)

    if original_size:
        input_img = im_gray
    else:
        width = 200
        r = float(width) / im_gray.shape[1]
        dim = (width, int(im_gray.shape[0] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(im_gray, dim, interpolation=cv2.INTER_AREA)
        input_img = resized

    valid_mask = input_img < 230
    orig_valid_mask = im_gray < 230

    graph = build_graph(input_img, valid_mask, debug=False)

    source = graph.get_vertex(0)
    target = graph.get_vertex(input_img.shape[1] - 1)
    # new code for shortest path with exactly k edges
    division_num, path, division_error = find_optimal_divisions_num(graph, source, target)
    print(path)
    if original_size:
        orig_path = path
    else:
        orig_path = np.int32(path / r)
        orig_path[-1] = (im_rgb.shape[1] - 1)
        # dijkstra old code
        # path = get_best_division(graph, source, target)

    if display:
        # print the division on the image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(im_rgb)
        # y_path = np.ones(x[path].shape) * y_size
        # ax.bar(x[path], y_path, '.-', color='green')
        ax.vlines(orig_path, ymin=0, ymax=im_rgb.shape[0], color='red')

        plt.show()

        # 3D display
        draw_3d_division(im_gray, orig_valid_mask, orig_path)

        return orig_path, division_error


if __name__ == "__main__":

    eye_img = cv2.imread(r'../images/src.jpg')
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    sectors = img_to_sectors(eye_img, center=(501, 273), radius=285, num_sectors=10, plot_debug=True)

    if False:
        slice_img = cv2.imread(r'../images/input.png')

        slice_path, slice_error = get_slice_partition(slice_img, display=True)

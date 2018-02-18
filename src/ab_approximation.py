import numpy as np
import math
import multiprocessing
from multiprocessing import Pool


class Point:
    def __init__(self, index, coordinates):
        self.coordinates = coordinates
        self.index = index


def k_means_ab_approx(points, k, eps, delta):
    k_means = []
    while len(points) > k:
        centers = gamma_eps_sample_by_vc(points, k, eps, delta)
        k_means += centers
        points_with_distances = get_points_with_distances(points, centers)
        number_of_points_to_remove = int(math.ceil(len(points) / 2.0))
        points = remove_closest_points(points, points_with_distances, number_of_points_to_remove)

    k_means += points
    for i in range(0, len(k_means), 1):
        k_means[i].index = i
    return k_means


def get_points_with_distances(points, centers):
    points_with_distances = []  # Points with distances from their center.
    cpu_count = multiprocessing.cpu_count()
    results = [None] * (cpu_count - 1)
    pool = Pool(processes=cpu_count - 1)
    points_for_pool_count = int(len(points) / (cpu_count - 1))
    for cpu in range(0, cpu_count - 2, 1):
        results[cpu] = pool.apply_async(calculate_distances_parallel, args=(points[points_for_pool_count * cpu: points_for_pool_count * (cpu + 1)], centers))
    results[cpu_count - 2] = pool.apply_async(calculate_distances_parallel, args=(points[points_for_pool_count * (cpu_count - 2):], centers))
    pool.close()
    pool.join()
    for result in results:
        points_with_distances.extend(result.get())
    return points_with_distances


def calculate_distances_parallel(points, centers):
    points_with_distances = []
    for point in points:
        points_with_distances.append([find_dist_from_centers(point, centers), point])
    return points_with_distances


def gamma_eps_sample_by_vc(points, k, eps, delta):
    vc = 3  # Should be set according to the problem classification
    gamma = 0.5
    size = int((1 / ((eps * gamma) ** 2)) * (vc * math.log(len(points), 2) + math.log(1 / delta, 2)))
    # The above line is the formula written in the theoretical articles. In practice the sample size
    # is astronomically high, especially for our data. Had to compromise and overwrite the size for the code to work in
    # a reasonable time.
    size = int(k * eps)  # Should be set according to the data received.
    if size > len(points):
        size = len(points)
    sample = np.random.choice(points, size, replace=False)
    return np.ndarray.tolist(sample)


def find_dist_from_centers(point, centers):
    dists = []
    for center in centers:
        dists.append(find_euclidean_distance(point, center))
    return reduce(find_min_dist, dists)


def remove_closest_points(points, points_with_distances, num_of_points_to_remove):
    points_with_distances.sort(key=lambda x: x[0])
    remaining_points = points_with_distances[num_of_points_to_remove:]
    points = map(lambda d: d[1], remaining_points)
    return points


def find_euclidean_distance(point, center):
    return np.linalg.norm(np.subtract(point.coordinates, center.coordinates))


def find_min_dist(dist1, dist2):
    if dist1 < dist2:
        return dist1
    else:
        return dist2

import numpy as np
import math
import sys
import multiprocessing
from multiprocessing import Pool
import ab_approximation as abApproximator
from ab_approximation import Point


class PointWithCenter:
    def __init__(self, index, coordinates, center, distance_from_center):
        self.coordinates = coordinates
        self.index = index
        self.center = center
        self.distance_from_center = distance_from_center


class CoresetPoint:
    def __init__(self, index, coordinates, weight):
        self.coordinates = coordinates
        self.index = index
        self.weight = weight


def k_means_coreset(p, weights, k, eps, delta):
    points_for_ab = []
    for i in range(0, len(p), 1):
        points_for_ab.append(Point(i, p[i]))
    ab_approx = abApproximator.k_means_ab_approx(points_for_ab, k, eps, delta)

    points, centers_cluster_sizes = calculate_centers_for_points(p, ab_approx)
    sensitivities = calculate_sensitivities(points, centers_cluster_sizes)
    probabilities = np.array(sensitivities) / sum(sensitivities)
    coreset_points = sample_by_probability(points, k, probabilities)
    coreset = calculate_weights_and_create_coreset_points(coreset_points, probabilities, weights)
    return coreset


def calculate_centers_for_points(p, ab_approx):
    # Finds the center for each point from the data and create the PointWithCenter class
    # Also calculates the center cluster size in the 'find_projection_of_point_in_ab_approx'
    points = []
    centers_cluster_sizes = [0] * len(ab_approx)
    cpu_count = multiprocessing.cpu_count()
    results = [None] * (cpu_count - 1)
    pool = Pool(processes=cpu_count - 1)
    points_for_pool_count = int(len(p) / (cpu_count - 1))
    for cpu in range(0, cpu_count - 2, 1):
        results[cpu] = pool.apply_async(calculate_distances, args=(p, ab_approx, (points_for_pool_count * cpu), (points_for_pool_count * (cpu + 1))))
    results[cpu_count - 2] = pool.apply_async(calculate_distances, args=(p, ab_approx, points_for_pool_count * (cpu_count - 2), len(p)))
    pool.close()
    pool.join()
    for result in results:
        points_result, cluster_sizes_result = result.get()
        points.extend(points_result)
        centers_cluster_sizes = np.add(centers_cluster_sizes, cluster_sizes_result)
    return points, centers_cluster_sizes


def calculate_distances(p, ab_approx, start_index, end_index):
    points = []
    centers_cluster_sizes = [0] * len(ab_approx)
    for i in range(start_index, end_index, 1):
        point_center, center_distance = find_projection_of_point_in_ab_approx(p[i], ab_approx, centers_cluster_sizes)
        points.append(PointWithCenter(i, p[i], point_center, center_distance))
    return points, centers_cluster_sizes


def calculate_sensitivities(points, centers_cluster_sizes):
    sensitivities = []
    distances_sum = calculate_total_sum_of_distances(points)
    if distances_sum == 0:
        return np.ones(len(points), dtype=np.float64)
        # When the sum is 0 it's because all the points are centers. In this case each cluster has one point.
    for i in range(0, len(points), 1):
        point_cluster_size = centers_cluster_sizes[points[i].center.index]
        sensitivities.append((points[i].distance_from_center / distances_sum) + (1.0 / point_cluster_size))
    return sensitivities


def calculate_weights_and_create_coreset_points(coreset_points, probabilities, weights):
    coreset = []
    for i in range(0, len(coreset_points), 1):
        # u(p) = w(p)/(|S|*Prob(p)) - The result weights mimic the actual data within the reasonable error (+-10%)
        coreset_point_weight = weights[coreset_points[i].index] / (probabilities[coreset_points[i].index] * len(coreset_points))
        coreset.append(CoresetPoint(i, coreset_points[i].coordinates, coreset_point_weight))
    return coreset


# Finds the center for a point in the ab-approximation. Also counts cluster sizes.
def find_projection_of_point_in_ab_approx(point, ab_approx, centers_cluster_sizes):
    min_dist = sys.maxint
    for point_approx in ab_approx:
        curr_dist = find_euclidean_distance(point, point_approx.coordinates)
        if curr_dist < min_dist:
            min_dist, closest_center = curr_dist, point_approx
    centers_cluster_sizes[closest_center.index] += 1
    return closest_center, min_dist


def calculate_total_sum_of_distances(points):
    sum_of_distances = 0
    for point in points:
        sum_of_distances += point.distance_from_center
    return sum_of_distances


def find_euclidean_distance(point, center):
    return np.linalg.norm(np.subtract(point, center))


def sample_by_probability(points, size, probability):
    sample = np.random.choice(points, size, p=probability, replace=False)
    return sample


def create(eye_img, height, width, k):
    points = []
    eps = 0.2
    delta = 3.0 / 4.0
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            points.append([i, j])
    weights = np.ones(len(points), dtype=np.float64)
    new_height = int(math.sqrt((float(eye_img.shape[0]) / eye_img.shape[1]) * k))
    new_width = int(k / new_height)
    result = k_means_coreset(points, weights, k, eps, delta)
    # sort according to square locations in matrix
    result.sort(key=lambda x: (int(x.coordinates[0] / new_height) * eye_img.shape[0]) + x.coordinates[1])
    coreset_eye_image = np.zeros((new_height, new_width), dtype=np.uint8)
    result_index = 0
    for i in range(0, new_height, 1):
        for j in range(0, new_width, 1):
            coreset_eye_image[i][j] = eye_img[result[result_index].coordinates[0]][result[result_index].coordinates[1]]
            result_index += 1

    return coreset_eye_image

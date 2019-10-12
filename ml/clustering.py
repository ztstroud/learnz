import numpy as np
import random


def euclidean_distance(vector_one, vector_two):
    """
    Determines the euclidean distance between two numpy vectors.

    :param vector_one: the first vector
    :param vector_two: the second vector

    :return: the distance between the given vectors
    """

    return np.linalg.norm(vector_two - vector_one)

def get_center(centers, vector, distance_function):
    """
    Determines which center is closest to the given vector.

    The given distance function will be used to calculate the distance between
    the vector and possible centers.

    The centers should be a list of numpy vectors, and vector should be a numpy
    vector.

    :param centers: the centers to check
    :param vector: the vector to check
    :param distance_function: the distance function to use

    :return: the center closest to the given vector
    """

    min_distance = None
    min_center = None

    for center in centers:
        distance = distance_function(center, vector)

        if min_distance is None or distance < min_distance:
            min_distance = distance
            min_center = center

    return min_center

def distance_to_center(centers, vector, distance_function):
    """
    Gets the distance between the given vector and its closest center.

    The given distance function will be used to calculate the distance between
    the vector and possible centers.

    The centers should be a list of numpy vectors, and vector should be a numpy
    vector.
    
    :param centers: the centers to check
    :param vector: the vector to check
    :param distance_function: the distance function to use

    :return: the distance between the given vector and its closest center
    """

    center = get_center(centers, vector, distance_function)
    return distance_function(center, vector)

def get_clusters(centers, data, distance_function):
    """
    Gets the clusters that correspond to the given centers.

    The given distance function will be used to calculate the distance between
    a pair of vectors.

    The centers should be a list of numpy vectors. data can be a list of numpy
    vectors or a numpy array of numpy vectors.

    The clusters will be returned as a list of clusters, each of which is
    represented as a list of numpy vectors.

    :param centers: centers to cluster around
    :param data: list of vectors to cluster
    :param distance_function: the distance function to use

    :return: the generated clusters
    """
    
    clusters = [[] for center in centers]

    for vector in data:
        center = get_center(centers, vector, distance_function)

        for index in range(len(centers)):
            if np.array_equal(center, centers[index]):
                clusters[index].append(vector)
                break

    return clusters

def gonzalez(data, center_count, distance_function = euclidean_distance, *, run_lloyds = True, return_centers = False):
    """
    The Gonzalez algorithm for k-means clustering.

    The first center is chosen arbitrarily as the first data point. The next
    center is selected as the vector that is furthest away from its center. This
    process is repeated until the desired number of centers are found.

    data can be a list of numpy vectors or a numpy array of numpy vectors.

    The given distance function will be used to calculate the distance between
    a pair of vectors.

    :param data: list of vectors to cluster
    :param center_count: the number of centers to find
    :param distance_function: the distance function to use (default is euclidean)
    :param run_lloyds: if True, run Lloyd's algorithm on the initial centers (default True)
    :param return_centers: if True, the centers of the clusters will be returned with the clusters

    :return: the clusters created by the gonzalez algorithm
    """

    centers = [data[0]]

    while len(centers) < center_count:
        max_distance = None
        max_vector = None

        for vector in data:
            distance = distance_to_center(centers, vector, distance_function)

            if max_distance is None or distance > max_distance:
                max_distance = distance
                max_vector = vector

        centers.append(max_vector)

    if run_lloyds:
        centers = lloyds_algorithm(centers, data, distance_function)

    clusters = get_clusters(centers, data, distance_function)

    if return_centers:
        return clusters, centers

    return clusters

def kmeans_pp(data, center_count, distance_function = euclidean_distance, *, run_lloyds = True, return_centers = False):
    """
    The k-means++ algorithm for k-means clustering.

    The first center is chosen arbitrarily as the first data point. The next
    center is selected randomly from all data points proportional to the square
    of its distance from its center. This process is repeated until the desired
    number of centers are found.

    data can be a list of numpy vectors or a numpy array of numpy vectors.

    The given distance function will be used to calculate the distance between a
    pair of vectors.

    :param data: list of vectors to cluster
    :param center_count: the number of centers to find
    :param distance_function: the distance function to use (default is euclidean)
    :param run_lloyds: if True, run Lloyd's algorithm on the initial centers (default True)
    :param return_centers: if True, the centers of the clusters will be returned with the clusters (default False)

    :return: the clusters created by the k-means++ algorithm
    """

    centers = [data[0]]

    while len(centers) < center_count:
        distances = [pow(distance_to_center(centers, vector, distance_function), 2) for vector in data]
        total_distance = sum(distances)

        distances = [distance / total_distance for distance in distances]

        uniform = random.uniform(0.0, 1.0)
        cumulative = 0.0

        for index, distance in enumerate(distances):
            cumulative += distance

            if uniform < cumulative:
                centers.append(data[index])
                break

    if run_lloyds:
        centers = lloyds_algorithm(centers, data, distance_function)

    clusters = get_clusters(centers, data, distance_function)

    if return_centers:
        return clusters, centers

    return clusters

def lloyds_algorithm(centers, data, distance_function):
    """
    Executes lloyd's algorithm and returns the new centers.

    data can be a list of numpy vectors or a numpy array of numpy vectors.

    The given distance function will be used to calculate the distance between a
    pair of vectors.

    :param centers: the centers to adjust
    :param data: list of vectors to cluster
    :param distance_function: the distance function to use

    :return: the adjusted clusters
    """

    while True:
        clusters = get_clusters(centers, data, distance_function)
        new_centers = [np.average(cluster, axis = 0) for cluster in clusters]

        changed = False
        for old_center, new_center in zip(centers, new_centers):
            if not np.array_equal(old_center, new_center):
                changed = True
                break

        if not changed:
            return centers

        centers = new_centers

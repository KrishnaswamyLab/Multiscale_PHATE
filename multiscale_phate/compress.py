import numpy as np
import joblib
import tasklogger
import sklearn.cluster
import sklearn.neighbors
import scipy.spatial.distance


_logger = tasklogger.get_tasklogger("graphtools")


def get_compression_features(N, features, n_pca, partitions):
    """Short summary. TODO

    Parameters
    ----------
    N : type TODO
        Description of parameter `N`. TODO
    features : type TODO
        Description of parameter `features`. TODO
    n_pca : type TODO
        Description of parameter `n_pca`. TODO
    partitions : type TODO
        Description of parameter `partitions`. TODO
    landmarks : type TODO
        Description of parameter `landmarks`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    if n_pca is None:
        n_pca = min(N, features)
    if n_pca > 100:
        n_pca = 100

    # if N<100000:
    #     partitions=None
    if partitions is not None and partitions >= N:
        partitions = None

    if partitions is not None and partitions > 50000:
        partitions = 50000
    elif N > 100000:
        partitions = 20000

    return n_pca, partitions


def cluster_components(data_subset, num_cluster, size, random_state=None):
    """Short summary. TODO

    Parameters
    ----------
    data_subset : type TODO
        Description of parameter `data_subset`. TODO
    num_cluster : type TODO
        Description of parameter `num_cluster`. TODO
    size : type TODO
        Description of parameter `size`. TODO
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize MiniBatchKMeans.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    if data_subset.shape[0] == 1:
        return [0]
    k = np.ceil(data_subset.shape[0] / size).astype(int)
    # print(data_subset.shape)
    # print(k)
    if k > num_cluster:
        k = num_cluster
    # print(k)
    mbk = sklearn.cluster.MiniBatchKMeans(
        init="k-means++",
        n_clusters=k,
        batch_size=k * 10,
        n_init=10,
        max_no_improvement=10,
        verbose=0,
        random_state=random_state,
    ).fit(data_subset)
    return mbk.labels_


def subset_data(data, desired_num_clusters, n_jobs, num_cluster=100, random_state=None):
    """Short summary. TODO

    Parameters
    ----------
    data : type TODO
        Description of parameter `data`. TODO
    desired_num_clusters : type TODO
        Description of parameter `desired_num_clusters`. TODO
    n_jobs : type TODO
        Description of parameter `n_jobs`. TODO
    num_cluster : type TODO
        Description of parameter `num_cluster`. TODO
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize MiniBatchKMeans.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    N = data.shape[0]
    size = int(N / desired_num_clusters)
    with _logger.task("partitions"):

        mbk = sklearn.cluster.MiniBatchKMeans(
            init="k-means++",
            n_clusters=num_cluster,
            batch_size=num_cluster * 10,
            n_init=10,
            max_no_improvement=10,
            verbose=0,
            random_state=random_state,
        ).fit(data)

        clusters = mbk.labels_
        clusters_unique, cluster_counts = np.unique(clusters, return_counts=True)
        clusters_next_iter = clusters.copy()

        while np.max(cluster_counts) > np.ceil(N / desired_num_clusters):
            min_val = 0
            partitions_id_uni = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(cluster_components)(
                    data[np.where(clusters == clusters_unique[i])[0], :],
                    num_cluster,
                    size,
                    random_state=random_state,
                )
                for i in range(len(clusters_unique))
            )

            for i in range(len(clusters_unique)):
                loc = np.where(clusters == clusters_unique[i])[0]
                clusters_next_iter[loc] = np.array(partitions_id_uni[i]) + min_val
                min_val = min_val + np.max(np.array(partitions_id_uni[i])) + 1

            clusters = clusters_next_iter.copy()
            clusters_unique, cluster_counts = np.unique(clusters, return_counts=True)

    return clusters


def merge_clusters(diff_pot_unmerged, clusters):
    """Short summary. TODO

    Parameters
    ----------
    diff_pot_unmerged : type TODO
        Description of parameter `diff_pot_unmerged`. TODO
    clusters : type TODO
        Description of parameter `clusters`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    clusters_uni = np.unique(clusters)
    num_clusters = len(clusters_uni)
    diff_pot_merged = np.zeros(num_clusters * diff_pot_unmerged.shape[1]).reshape(
        num_clusters, diff_pot_unmerged.shape[1]
    )

    for c in range(num_clusters):
        loc = np.where(clusters_uni[c] == clusters)[0]
        diff_pot_merged[c, :] = np.nanmean(diff_pot_unmerged[loc], axis=0)

    return diff_pot_merged


def get_distance_from_centroids(centroids, data, clusters):
    """Short summary.

    Parameters
    ----------
    centroids : type
        Description of parameter `centroids`.
    data : type
        Description of parameter `data`.
    clusters : type
        Description of parameter `clusters`.

    Returns
    -------
    type
        Description of returned object.
    """
    distance = np.zeros(centroids.shape[0])

    for c in range(centroids.shape[0]):
        cluster_points = data[clusters == c]
        dist = []

        for i in range(cluster_points.shape[0]):
            dist.append(
                scipy.spatial.distance.sqeuclidean(
                    centroids[c, :], cluster_points[i, :]
                )
            )
        distance[c] = np.max(dist)
    return distance


def map_update_data(centroids, data, new_data, partition_clusters, nn=5, n_jobs=10):
    """Short summary. TODO

    Parameters
    ----------
    centroids : type TODO
        Description of parameter `centroids`. TODO
    data : type TODO
        Description of parameter `data`. TODO
    new_data : type TODO
        Description of parameter `new_data`. TODO
    partition_clusters : type TODO
        Description of parameter `partition_clusters`. TODO
    nn : type TODO
        Description of parameter `nn`. TODO
    n_jobs : type TODO
        Description of parameter `n_jobs`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    with _logger.task("map to computed partitions"):
        # getting max distance to each partition centroid
        distance_merged = get_distance_from_centroids(
            centroids, data, partition_clusters
        )

        # Mapping NN in new data to centroids
        NN_op = sklearn.neighbors.NearestNeighbors(n_neighbors=nn, n_jobs=n_jobs)
        NN_op.fit(centroids)
        neighbor_dists, neighbor_idx = NN_op.kneighbors(new_data)

        # Identifying which new data points fall below threshold
        parition_assignment_bool = neighbor_dists < distance_merged[neighbor_idx]

        subset_partition_assignment = np.zeros(new_data.shape[0])
        subset_partition_assignment[subset_partition_assignment == 0] = -1

        # Finding neatest mapped partition centroid
        for r in range(len(subset_partition_assignment)):
            c = 0
            while c < nn:
                if parition_assignment_bool[r, c] is True:
                    subset_partition_assignment[r] = neighbor_idx[r, c]
                    c = nn + 1
                    break
                else:
                    c += 1

    return subset_partition_assignment

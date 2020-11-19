import numpy as np
import phate
import tasklogger

_logger = tasklogger.get_tasklogger("graphtools")


def repulsion(temp):
    """Short summary. TODO

    Parameters
    ----------
    temp : type TODO
        Description of parameter `temp`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    for r in range(temp.shape[0]):
        for c in range(temp.shape[1]):
            val = temp[r, c]
            neg = val < 0
            if neg:
                temp[r, c] = -1 * np.sqrt(np.abs(val))
            else:
                temp[r, c] = np.sqrt(val)
    return temp


def condense_visualization(merge_pairs, phate):
    """Short summary. TODO

    Parameters
    ----------
    merge_pairs : type TODO
        Description of parameter `merge_pairs`. TODO
    phate : type TODO
        Description of parameter `phate`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    to_delete = []
    for m in range(len(merge_pairs)):
        to_merge = merge_pairs[m]
        phate[to_merge[0]] = np.mean(phate[to_merge], axis=0)
        to_delete.extend(to_merge[1:])
    phate = np.delete(phate, to_delete, axis=0)
    return phate


def compute_gradient(Xs, merges):
    """Short summary. TODO

    Parameters
    ----------
    Xs : type TODO
        Description of parameter `Xs`. TODO
    merges : type TODO
        Description of parameter `merges`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    _logger.info("Computing gradient...")
    gradient = []
    m = 0
    X = Xs[0]

    for layer in range(0, len(Xs) - 1):
        if X.shape[0] != Xs[layer + 1].shape[0]:
            X_1 = condense_visualization(merges[m], X)
            m = m + 1
            while X_1.shape[0] != Xs[layer + 1].shape[0]:
                X_1 = condense_visualization(merges[m], X_1)
                m = m + 1
        else:
            X_1 = X
        gradient.append(np.sum(np.abs(X_1 - Xs[layer + 1])))
        X = Xs[layer + 1]
    return np.array(gradient)


def get_levels(grad):
    """Short summary. TODO

    Parameters
    ----------
    grad : type TODO
        Description of parameter `Xs`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO
    """
    _logger.info("Identifying salient levels of resolution...")
    minimum = np.max(grad)
    levels = []
    levels.append(0)

    for i in range(1, len(grad) - 1):
        if grad[i] <= minimum and grad[i] < grad[i + 1]:
            levels.append(i)
            minimum = grad[i]
    return levels


def get_zoom_visualization(
    Xs,
    NxTs,
    zoom_visualization_level,
    zoom_cluster_level,
    coarse_cluster_level,
    coarse_cluster,
    n_jobs,
    random_state=None,
):
    """Short summary TODO

    Parameters
    ----------
    TODO
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize MDS.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator
    """
    unique = np.unique(
        NxTs[zoom_visualization_level], return_index=True, return_counts=True
    )
    extract = NxTs[coarse_cluster_level][unique[1]] == coarse_cluster

    subset_X = Xs[zoom_visualization_level]
    embedding = phate.mds.embed_MDS(subset_X[extract], n_jobs=n_jobs, seed=random_state)

    return embedding, NxTs[zoom_cluster_level][unique[1]][extract], unique[2][extract]


def compute_ideal_visualization_layer(gradient, Xs, min_cells=100):
    """Short summary. TODO

    Parameters
    ----------
    gradient : type TODO
        Description of parameter `gradient`. TODO
    Xs : type TODO
        Description of parameter `Xs`. TODO
    min_cells : type TODO
        Description of parameter `min_cells`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO
    """
    minimum = np.max(gradient)
    min_layer = 0

    for layer in range(1, len(Xs)):
        if Xs[layer].shape[0] < min_cells:
            break
        if gradient[layer] < minimum:
            # print("New minimum!")
            minimum = gradient[layer]
            min_layer = layer
    return min_layer


def get_clusters_sizes_2(
    clusters_full, layer, NxT, X, repulse=False, n_jobs=10, random_state=None
):
    """Short summary. TODO

    Parameters
    ----------
    clusters_full : type TODO
        Description of parameter `clusters_full`. TODO
    layer : type TODO
        Description of parameter `layer`. TODO
    NxT : type TODO
        Description of parameter `NxT`. TODO
    X : type TODO
        Description of parameter `X`. TODO
    repulse : type TODO
        Description of parameter `repulse`. TODO
    n_jobs : type TODO
        Description of parameter `n_jobs`. TODO
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize MDS.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type TODO
        Description of returned object. TODO
    """
    unique = np.unique(NxT[layer], return_index=True, return_counts=True)

    # expand_X = Xs[layer][scale_down(NxTs[layer])]
    # subset_X = expand_X[np.unique(NxTs[layer], return_index=True)[1]]

    subset_X = X[layer]

    if repulse:
        subset_X = repulsion(subset_X.copy())

    embedding = phate.mds.embed_MDS(subset_X, n_jobs=n_jobs, seed=random_state)
    return embedding, clusters_full[unique[1]], unique[2]

import numpy as np
import phate
import tasklogger


def repulsion(temp):
    """Short summary.

    Parameters
    ----------
    temp : type
        Description of parameter `temp`.

    Returns
    -------
    type
        Description of returned object.

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
    """Short summary.

    Parameters
    ----------
    merge_pairs : type
        Description of parameter `merge_pairs`.
    phate : type
        Description of parameter `phate`.

    Returns
    -------
    type
        Description of returned object.

    """
    to_delete = []
    for m in range(len(merge_pairs)):
        to_merge = merge_pairs[m]
        phate[to_merge[0]] = np.mean(phate[to_merge], axis=0)
        to_delete.extend(to_merge[1:])
    phate = np.delete(phate, to_delete, axis=0)
    return phate


def compute_gradient(Xs, merges):
    """Short summary.

    Parameters
    ----------
    Xs : type
        Description of parameter `Xs`.
    merges : type
        Description of parameter `merges`.

    Returns
    -------
    type
        Description of returned object.

    """
    tasklogger.log_info("Computing gradient...")
    gradient = []
    m = 0
    X = Xs[0]

    for l in range(0, len(Xs) - 1):
        if X.shape[0] != Xs[l + 1].shape[0]:
            X_1 = condense_visualization(merges[m], X)
            m = m + 1
            while X_1.shape[0] != Xs[l + 1].shape[0]:
                X_1 = condense_visualization(merges[m], X_1)
                m = m + 1
        else:
            X_1 = X
        gradient.append(np.sum(np.abs(X_1 - Xs[l + 1])))
        X = Xs[l + 1]
    return np.array(gradient)


def get_levels(grad):
    """Short summary.

    Parameters
    ----------
    grad : type
        Description of parameter `Xs`.

    Returns
    -------
    type
        Description of returned object.


    """
    tasklogger.log_info("Identifying salient levels of resolution...")
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
    """Short summary

    Parameters
    ----------

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
    """Short summary.

    Parameters
    ----------
    gradient : type
        Description of parameter `gradient`.
    Xs : type
        Description of parameter `Xs`.
    min_cells : type
        Description of parameter `min_cells`.

    Returns
    -------
    type
        Description of returned object.

    """
    minimum = np.max(gradient)
    min_layer = 0

    for l in range(1, len(Xs)):
        if Xs[l].shape[0] < min_cells:
            break
        if gradient[l] < minimum:
            # print("New minimum!")
            minimum = gradient[l]
            min_layer = l
    return min_layer


def get_clusters_sizes_2(
    clusters_full, layer, NxT, X, repulse=False, n_jobs=10, random_state=None
):
    """Short summary.

    Parameters
    Parameters
    ----------
    clusters_full : type
        Description of parameter `clusters_full`.
    layer : type
        Description of parameter `layer`.
    NxT : type
        Description of parameter `NxT`.
    X : type
        Description of parameter `X`.
    repulse : type
        Description of parameter `repulse`.
    n_jobs : type
        Description of parameter `n_jobs`.
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize MDS.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type
        Description of returned object.

    """
    unique = np.unique(NxT[layer], return_index=True, return_counts=True)

    # expand_X = Xs[layer][scale_down(NxTs[layer])]
    # subset_X = expand_X[np.unique(NxTs[layer], return_index=True)[1]]

    subset_X = X[layer]

    if repulse:
        subset_X = repulsion(subset_X.copy())

    embedding = phate.mds.embed_MDS(subset_X, n_jobs=n_jobs, seed=random_state)
    return embedding, clusters_full[unique[1]], unique[2]

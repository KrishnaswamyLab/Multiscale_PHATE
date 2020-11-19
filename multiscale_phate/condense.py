import numpy as np

import graphtools
import tasklogger
import collections

import scipy.spatial.distance
import sklearn.metrics.pairwise

_logger = tasklogger.get_tasklogger("graphtools")


def comp(node, neigh, visited):
    """Short summary. TODO

    Parameters
    ----------
    node : type TODO
        Description of parameter `node`. TODO
    neigh : type TODO
        Description of parameter `neigh`. TODO
    visited : type TODO
        Description of parameter `visited`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    vis = visited.add
    nodes = set([node])
    next_node = nodes.pop
    while nodes:
        node = next_node()
        vis(node)
        nodes |= neigh[node] - visited
        yield node


def merge_common(lists):
    """Short summary. TODO

    Parameters
    ----------
    lists : type TODO
        Description of parameter `lists`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    neigh = collections.defaultdict(set)
    visited = set()
    for each in lists:
        for item in each:
            neigh[item].update(each)

    for node in neigh:
        if node not in visited:
            yield sorted(comp(node, neigh=neigh, visited=visited))


def compute_condensation_param(X, granularity):
    """Short summary. TODO

    Parameters
    ----------
    X : type TODO
        Description of parameter `X`. TODO
    granularity : type TODO
        Description of parameter `granularity`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    epsilon = granularity * (0.1 * np.mean(np.std(X))) / (X.shape[0] ** (-1 / 5))
    D = scipy.spatial.distance.pdist(X, metric="euclidean")
    merge_threshold = np.percentile(D, 0.001) + 0.001
    _logger.info("Setting epsilon to " + str(round(epsilon, 4)))
    _logger.info("Setting merge threshold to " + str(round(merge_threshold, 4)))
    return epsilon, merge_threshold


def condense(X, clusters, scale, epsilon, merge_threshold, n_jobs, random_state=None):
    """Short summary. TODO

    Parameters
    ----------
    X : type TODO
        Description of parameter `X`. TODO
    clusters : type TODO
        Description of parameter `clusters`. TODO
    scale : type TODO
        Description of parameter `scale`. TODO
    epsilon : type TODO
        Description of parameter `epsilon`. TODO
    merge_threshold : type TODO
        Description of parameter `merge_threshold`. TODO
    n_jobs : type TODO
        Description of parameter `n_jobs`. TODO
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize graphtools.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    NxT = []
    NxT.append(clusters)
    NxT.append(clusters)
    X_cont = []

    N = X.shape[0]

    for c in range(len(np.unique(clusters))):
        loc = np.where(c == clusters)[0]
        X_cont.append(list(loc))
    X_1 = X.copy()
    K_list = []
    X_list = []
    X_list.append(X_1)
    X_list.append(X_1)
    P_list = []
    merged = []
    with _logger.task("condensation"):
        while X_1.shape[0] > 1:
            D = sklearn.metrics.pairwise.pairwise_distances(
                X_1, metric="euclidean", n_jobs=n_jobs
            )
            bool_ = D < merge_threshold
            loc = np.where(bool_)
            merge_pairs = []
            for i in range(len(loc[0])):
                if loc[0][i] != loc[1][i]:
                    merge_pairs.append(tuple([loc[0][i], loc[1][i]]))

            while len(merge_pairs) == 0:
                epsilon = scale * epsilon
                G = graphtools.Graph(
                    X_1,
                    knn=min(X_1.shape[0] - 2, 5),
                    bandwidth=epsilon,
                    n_jobs=n_jobs,
                    random_state=random_state,
                )

                P_s = G.P.toarray()
                X_2 = P_s @ X_1

                X_1 = X_2
                X_list.append(X_1)
                P_list.append(P_s)
                NxT.append(NxT[-1].copy())
                K_list.append(G.K.toarray())

                D = sklearn.metrics.pairwise.pairwise_distances(
                    X_1, metric="euclidean", n_jobs=n_jobs
                )
                bool_ = D < merge_threshold
                loc = np.where(bool_)
                merge_pairs = []
                for i in range(len(loc[0])):
                    if loc[0][i] != loc[1][i]:
                        merge_pairs.append(tuple([loc[0][i], loc[1][i]]))

            merge_pairs = list(merge_common(merge_pairs))
            merged.append(merge_pairs)

            cluster_assignment = NxT[-1].copy()

            to_delete = []
            for m in range(len(merge_pairs)):
                to_merge = merge_pairs[m]
                X_1[to_merge[0]] = np.mean(X_1[to_merge], axis=0)
                to_delete.extend(to_merge[1:])
                for c in range(1, len(to_merge)):
                    X_cont[to_merge[0]].extend(X_cont[to_merge[c]])
                    cluster_assignment[X_cont[to_merge[c]]] = cluster_assignment[
                        X_cont[to_merge[0]]
                    ][0]

            X_1 = np.delete(X_1, to_delete, axis=0)
            X_cont = list(
                np.delete(np.asarray(X_cont, dtype=object), to_delete, axis=0)
            )

            del X_list[-1]
            X_list.append(X_1)

            del NxT[-1]
            NxT.append(cluster_assignment)

    return NxT, X_list, K_list, merged, P_list

import numpy as np
import phate


def repulsion(temp):
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
    to_delete = []
    for m in range(len(merge_pairs)):
        to_merge = merge_pairs[m]
        phate[to_merge[0]] = np.mean(phate[to_merge], axis=0)
        to_delete.extend(to_merge[1:])
    phate = np.delete(phate, to_delete, axis=0)
    return phate


def compute_gradient(Xs, merges):
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


def compute_ideal_visualization_layer(gradient, Xs, min_cells=100):
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


def get_clusters_sizes_2(clusters_full, layer, NxT, X, repulse=False, n_jobs=10):
    unique = np.unique(NxT[layer], return_index=True, return_counts=True)

    # expand_X = Xs[layer][scale_down(NxTs[layer])]
    # subset_X = expand_X[np.unique(NxTs[layer], return_index=True)[1]]

    subset_X = X[layer]

    if repulse:
        embedding = phate.mds.embed_MDS(repulsion(subset_X.copy()), n_jobs=n_jobs)
    else:
        embedding = phate.mds.embed_MDS(subset_X, n_jobs=n_jobs)
    return embedding, clusters_full[unique[1]], unique[2]

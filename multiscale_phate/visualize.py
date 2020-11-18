import numpy as np
import tasklogger
import warnings

from . import embed


def get_visualization(
    Xs, NxTs, cluster_level, visualization_level, repulse, random_state=None
):
    """Short summary.

    Parameters
    ----------
    Xs : type
        Description of parameter `Xs`.
    NxTs : type
        Description of parameter `NxTs`.
    merges : type
        Description of parameter `merges`.
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize MDS.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type
        Description of returned object.

    """
    (hp_embedding, cluster_viz, sizes_viz,) = embed.get_clusters_sizes_2(
        np.array(NxTs[cluster_level]),
        visualization_level,
        NxTs,
        Xs,
        repulse=repulse,
        random_state=random_state,
    )
    return hp_embedding, cluster_viz, sizes_viz


def build_visualization(Xs, NxTs, merges, gradient, min_cells, random_state=None):
    """Short summary.

    Parameters
    ----------
    Xs : type
        Description of parameter `Xs`.
    NxTs : type
        Description of parameter `NxTs`.
    merges : type
        Description of parameter `merges`.
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize MDS.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type
        Description of returned object.

    """

    min_layer = embed.compute_ideal_visualization_layer(gradient, Xs, min_cells)
    (hp_embedding, cluster_viz, sizes_viz,) = embed.get_clusters_sizes_2(
        np.array(NxTs[-35]),
        min_layer,
        NxTs,
        Xs,
        repulse=False,
        random_state=random_state,
    )
    return hp_embedding, cluster_viz, sizes_viz


def map_clusters_to_tree(clusters, NxTs):
    clusters_tree = []

    for l in range(len(NxTs) - 1):
        _, ind = np.unique(NxTs[l], return_index=True)
        clusters_tree.extend(clusters[ind])

    return clusters_tree


def build_condensation_tree(data_pca, diff_op, NxT, merged_list, Ps):
    """Short summary.

    Parameters
    ----------
    data_pca : type
        Description of parameter `data_pca`.
    diff_op : type
        Description of parameter `diff_op`.
    NxT : type
        Description of parameter `NxT`.
    merged_list : type
        Description of parameter `merged_list`.
    Ps : type
        Description of parameter `Ps`.

    Returns
    -------
    type
        Description of returned object.

    """
    with tasklogger.log_task("base visualization"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="Pre-fit PHATE should not be used to transform a new data matrix. "
                "Please fit PHATE to the new data by running 'fit' with the new data.",
            )
            tree_phate = diff_op.transform(data_pca)

    # tree_phate = Ps[0] @ tree_phate
    embeddings = []
    embeddings.append(
        np.concatenate(
            [
                tree_phate,
                np.repeat(0, tree_phate.shape[0]).reshape(tree_phate.shape[0], 1),
            ],
            axis=1,
        )
    )

    m = 0

    with tasklogger.log_task("tree"):
        for l in range(0, len(Ps)):
            if len(np.unique(NxT[l])) != len(np.unique(NxT[l + 1])):
                tree_phate_1 = embed.condense_visualization(merged_list[m], tree_phate)
                m = m + 1
            if Ps[l].shape[0] != tree_phate_1.shape[0]:
                tree_phate_1 = embed.condense_visualization(
                    merged_list[m], tree_phate_1
                )
                m = m + 1
            tree_phate = Ps[l] @ tree_phate_1
            embeddings.append(
                np.concatenate(
                    [
                        tree_phate,
                        np.repeat(l + 1, tree_phate.shape[0]).reshape(
                            tree_phate.shape[0], 1
                        ),
                    ],
                    axis=1,
                )
            )
        tree = np.vstack((embeddings))
    return tree

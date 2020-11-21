import numpy as np
import tasklogger
import warnings

from . import embed

_logger = tasklogger.get_tasklogger("graphtools")


def get_visualization(
    Xs, NxTs, cluster_level, visualization_level, repulse, random_state=None
):
    """Short summary. TODO

    Parameters
    ----------
    Xs : type TODO
        Description of parameter `Xs`.
    NxTs : type TODO
        Description of parameter `NxTs`.
    merges : type TODO
        Description of parameter `merges`.
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize MDS.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type TODO
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
    """Short summary. TODO

    Parameters
    ----------
    Xs : type TODO
        Description of parameter `Xs`.
    NxTs : type TODO
        Description of parameter `NxTs`.
    merges : type TODO
        Description of parameter `merges`.
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize MDS.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type TODO
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
    """Short summary.

    Parameters
    ----------
    clusters : type
        Description of parameter `clusters`.
    NxTs : type
        Description of parameter `NxTs`.

    Returns
    -------
    type
        Description of returned object.

    """
    clusters_tree = []

    for layer in range(len(NxTs) - 1):
        _, ind = np.unique(NxTs[layer], return_index=True)
        clusters_tree.extend(clusters[ind])

    return clusters_tree


def build_condensation_tree(data_pca, diff_op, NxT, merged_list, Ps):
    """Short summary. TODO

    Parameters
    ----------
    data_pca : type TODO
        Description of parameter `data_pca`.
    diff_op : type TODO
        Description of parameter `diff_op`.
    NxT : type TODO
        Description of parameter `NxT`.
    merged_list : type TODO
        Description of parameter `merged_list`.
    Ps : type TODO
        Description of parameter `Ps`.

    Returns
    -------
    type TODO
        Description of returned object.

    """
    with _logger.task("base visualization"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="Pre-fit PHATE should not be used to transform a new data "
                "matrix. Please fit PHATE to the new data by running 'fit' with the "
                "new data.",
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

    with _logger.task("tree"):
        for layer in range(0, len(Ps)):
            if len(np.unique(NxT[layer])) != len(np.unique(NxT[layer + 1])):
                tree_phate_1 = embed.condense_visualization(merged_list[m], tree_phate)
                m = m + 1
            if Ps[layer].shape[0] != tree_phate_1.shape[0]:
                tree_phate_1 = embed.condense_visualization(
                    merged_list[m], tree_phate_1
                )
                m = m + 1
            tree_phate = Ps[layer] @ tree_phate_1
            embeddings.append(
                np.concatenate(
                    [
                        tree_phate,
                        np.repeat(layer + 1, tree_phate.shape[0]).reshape(
                            tree_phate.shape[0], 1
                        ),
                    ],
                    axis=1,
                )
            )
        tree = np.vstack((embeddings))
    return tree

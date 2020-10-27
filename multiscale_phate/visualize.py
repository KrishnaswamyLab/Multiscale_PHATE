import numpy as np
import tasklogger

from . import embed


def build_visualization(Xs, NxTs, merges):
    gradient = embed.compute_gradient(Xs, merges)
    min_layer = embed.compute_ideal_visualization_layer(gradient, Xs, min_cells=1000)
    (hp_embedding, cluster_viz, sizes_viz,) = embed.get_clusters_sizes_2(
        np.array(NxTs[-35]), min_layer, NxTs, Xs, repulse=False
    )
    return hp_embedding, cluster_viz, sizes_viz


def build_condensation_tree(data_pca, diff_op, NxT, merged_list, Ps):
    with tasklogger.log_task("base visualization"):
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

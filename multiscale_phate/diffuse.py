import numpy as np
import tasklogger
import phate
import sklearn.decomposition

from . import compress


def compute_diffusion_potential(data, N, decay, gamma, knn, landmarks=2000, n_jobs=10):
    with tasklogger.log_task("diffusion potential"):

        if landmarks != None and landmarks > data.shape[0]:
            landmarks = None

        diff_op = phate.PHATE(
            verbose=False,
            n_landmark=landmarks,
            n_pca=None,
            decay=decay,
            gamma=gamma,
            knn=knn,
            n_jobs=n_jobs,
        )
        diff_op.fit(data)

        pca = sklearn.decomposition.PCA(n_components=25)
        diff_potential_pca = pca.fit_transform(diff_op.diff_potential)

    return (
        diff_potential_pca[
            :, pca.explained_variance_ / np.sum(pca.explained_variance_) > 0.01
        ],
        diff_op,
        pca,
    )


def online_update_diffusion_potential(unmapped_data, diff_op, dp_pca):
    with tasklogger.log_task("extended diffusion potential"):
        with tasklogger.log_task("extended kernel"):
            # Extending kernel to new data
            transitions = diff_op.graph.extend_to_data(unmapped_data)

        try:
            merged_dp = compress.merge_clusters(
                diff_op.diff_potential, diff_op.graph.clusters
            )
        except AttributeError:
            # not a landmarkgraph
            merged_dp = diff_op.diff_potential

        dp_full = np.concatenate(
            (diff_op.diff_potential, (transitions.toarray() @ merged_dp)), axis=0
        )

        new_diff_potential_pca = dp_pca.transform(dp_full)

    return new_diff_potential_pca[
        :, dp_pca.explained_variance_ / np.sum(dp_pca.explained_variance_) > 0.01
    ]

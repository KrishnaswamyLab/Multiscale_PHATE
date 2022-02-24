import numpy as np
import tasklogger
import sklearn.decomposition
from . import compress, diffuse, condense


def build_tree(
    data_input,
    scale=1.025,
    landmarks=1000,
    partitions=None,
    granularity=0.1,
    n_pca=None,
    decay=40,
    gamma=1,
    knn=5,
    n_jobs=10,
    random_state=None,
):
    """Short summary.
    Parameters
    ----------
    data_input : type
        Description of parameter `data_input`.
    scale : type
        Description of parameter `scale`.
    landmarks : type
        Description of parameter `landmarks`.
    partitions : type
        Description of parameter `partitions`.
    granularity : type
        Description of parameter `granularity`.
    n_pca : type
        Description of parameter `n_pca`.
    decay : type
        Description of parameter `decay`.
    gamma : type
        Description of parameter `gamma`.
    knn : type
        Description of parameter `knn`.
    n_jobs : type
        Description of parameter `n_jobs`.
    random_state : integer or numpy.RandomState, optional, default: None
        The random number generator.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator
    Returns
    -------
    type
        Description of returned object.
    """
    with tasklogger.log_task("Multiscale PHATE tree"):
        N, features = data_input.shape

        # Computing compression features
        n_pca, partitions = compress.get_compression_features(
            N, features, n_pca, partitions, landmarks
        )

        with tasklogger.log_task("PCA"):
            pca_op = sklearn.decomposition.PCA(n_components=n_pca, random_state = random_state)
            data_pca = pca_op.fit_transform(np.array(data_input))
        clusters = np.arange(N)

        # Subsetting if required
        if partitions != None:
            partition_clusters = compress.subset_data(
                data_pca, partitions, n_jobs=n_jobs, random_state=random_state
            )
            data_pca = compress.merge_clusters(data_pca, partition_clusters)
            clusters = partition_clusters

        X, diff_op, diff_pca = diffuse.compute_diffusion_potential(
            data_pca, N, decay, gamma, knn, landmarks, n_jobs, random_state=random_state
        )

        epsilon, merge_threshold = condense.compute_condensation_param(
            X, granularity=granularity
        )

        NxTs, Xs, Ks, Merges, Ps = condense.condense(
            X,
            clusters,
            scale,
            epsilon,
            merge_threshold,
            n_jobs,
            random_state=random_state,
        )

    return (
        NxTs,
        Xs,
        Ks,
        Merges,
        Ps,
        diff_op,
        data_pca,
        pca_op,
        clusters,
        diff_pca,
        epsilon,
        merge_threshold,
    )

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
            pca_op = sklearn.decomposition.PCA(n_components=n_pca)
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


def online_update_tree(
    data_1,
    data_2,
    pca_centroid,
    pca_op,
    partitions,
    diff_operator,
    diff_pca_op,
    Xs,
    NxTs,
    Ks,
    Merges,
    Ps,
    scale,
    n_jobs=10,
    random_state=None,
):
    """Short summary.

    Parameters
    ----------
    data_1 : type
        Description of parameter `data_1`.
    data_2 : type
        Description of parameter `data_2`.
    pca_centroid : type
        Description of parameter `pca_centroid`.
    pca_op : type
        Description of parameter `pca_op`.
    partitions : type
        Description of parameter `partitions`.
    diff_operator : type
        Description of parameter `diff_operator`.
    diff_pca_op : type
        Description of parameter `diff_pca_op`.
    Xs : type
        Description of parameter `Xs`.
    NxTs : type
        Description of parameter `NxTs`.
    Ks : type
        Description of parameter `Ks`.
    Merges : type
        Description of parameter `Merges`.
    Ps : type
        Description of parameter `Ps`.
    scale : type
        Description of parameter `scale`.
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
    with tasklogger.log_task("Multiscale PHATE tree mapping"):
        if data_1.shape[0] != len(np.unique(partitions)):
            tasklogger.log_info("PCA compressing new data...")
            data_pca_1 = pca_op.transform(np.array(data_1))
            data_pca_2 = pca_op.transform(np.array(data_2))

            # Mapping new data to partitions
            partition_assignments = compress.map_update_data(
                pca_centroid, data_pca_1, data_pca_2, partitions, nn=5, n_jobs=n_jobs
            )
            tasklogger.log_info(
                "Points not mapped to partitions: "
                + str(sum(partition_assignments == -1))
            )

            # creating new joint paritions mapping
            new_partition_clusters = list(partitions)

            new_partition_clusters.extend(partition_assignments)
            new_partition_clusters = np.asarray(new_partition_clusters)

            update_idx = np.where(new_partition_clusters == -1)[0]

            max_partition = max(new_partition_clusters)

            for i in range(len(update_idx)):
                new_partition_clusters[update_idx[i]] = max_partition + 1
                max_partition += 1

            if sum(partition_assignments == -1) > 0:
                diff_pot_1 = diffuse.online_update_diffusion_potential(
                    data_pca_2[partition_assignments == -1, :],
                    diff_operator,
                    diff_pca_op,
                )
                epsilon, merge_threshold = condense.compute_condensation_param(
                    diff_pot_1, granularity=0.1
                )  # change to granularity

                pca_total = np.concatenate(
                    [pca_centroid, data_pca_2[partition_assignments == -1, :]]
                )

                NxTs_n, Xs_n, Ks_n, Merges_n, Ps_n = condense.condense(
                    diff_pot_1,
                    new_partition_clusters,
                    scale,
                    epsilon,
                    merge_threshold,
                    n_jobs=n_jobs,
                    random_state=random_state,
                )
                return NxTs_n, Xs_n, Ks_n, Merges_n, Ps_n, pca_total

            else:
                clusters = new_partition_clusters
                tasklogger.log_info("Rebuilding condensation tree...")
                clusters_idx = []

                for c in clusters:
                    clusters_idx.append(np.where(NxTs[0] == c)[0][0])

                NxTs_l = []

                for l in range(len(NxTs)):
                    NxTs_l.append(NxTs[l][clusters_idx])
                return NxTs_l, Xs, Ks, Merges, Ps, pca_centroid

        else:
            tasklogger.log_info("PCA compressing new data...")
            data_pca_2 = pca_op.transform(np.array(data_2))
            diff_pot_1 = diffuse.online_update_diffusion_potential(
                data_pca_2, diff_operator, diff_pca_op
            )
            clusters = np.arange(diff_pot_1.shape[0])

            epsilon, merge_threshold = condense.compute_condensation_param(
                diff_pot_1, granularity=0.1
            )  # change to granularity

            NxTs_n, Xs_n, Ks_n, Merges_n, Ps_n = condense.condense(
                diff_pot_1,
                clusters,
                scale,
                epsilon,
                merge_threshold,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            return (
                NxTs_n,
                Xs_n,
                Ks_n,
                Merges_n,
                Ps_n,
                np.concatenate([pca_centroid, data_pca_2]),
            )

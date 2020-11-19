import numpy as np
import tasklogger
import phate
import sklearn.decomposition

from . import compress

_logger = tasklogger.get_tasklogger("graphtools")


def compute_diffusion_potential(
    data, N, decay, gamma, knn, landmarks=2000, n_jobs=10, verbose=0, random_state=None
):
    """Short summary. TODO

    Parameters
    ----------
    data : type TODO
        Description of parameter `data`. TODO
    N : type TODO
        Description of parameter `N`. TODO
    decay : type TODO
        Description of parameter `decay`. TODO
    gamma : type TODO
        Description of parameter `gamma`. TODO
    knn : type TODO
        Description of parameter `knn`. TODO
    landmarks : type TODO
        Description of parameter `landmarks`. TODO
    n_jobs : type TODO
        Description of parameter `n_jobs`. TODO
    verbose : `int`, optional (default: 0)
        If `> 0`, print status messages
    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize PHATE and PCA.
        If an integer is given, it fixes the seed.
        Defaults to the global `numpy` random number generator

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    with _logger.task("diffusion potential"):

        if landmarks != None and landmarks > data.shape[0]:
            landmarks = None

        diff_op = phate.PHATE(
            n_landmark=landmarks,
            decay=decay,
            gamma=gamma,
            n_pca=None,
            knn=knn,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        diff_op.fit(data)

        pca = sklearn.decomposition.PCA(n_components=25, random_state=random_state)
        diff_potential_pca = pca.fit_transform(diff_op.diff_potential)

    return (
        diff_potential_pca[
            :, pca.explained_variance_ / np.sum(pca.explained_variance_) > 0.01
        ],
        diff_op,
        pca,
    )


def online_update_diffusion_potential(unmapped_data, diff_op, dp_pca):
    """Short summary. TODO

    Parameters
    ----------
    unmapped_data : type TODO
        Description of parameter `unmapped_data`. TODO
    diff_op : type TODO
        Description of parameter `diff_op`. TODO
    dp_pca : type TODO
        Description of parameter `dp_pca`. TODO

    Returns
    -------
    type TODO
        Description of returned object. TODO

    """
    with _logger.task("extended diffusion potential"):
        with _logger.task("extended kernel"):
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

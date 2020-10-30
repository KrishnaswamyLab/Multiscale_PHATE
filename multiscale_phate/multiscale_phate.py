from . import tree, embed, utils, visualize


class Multiscale_PHATE(object):
    """Short summary.

    Parameters
    ----------
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

    Attributes
    ----------
    scale
    landmarks
    partitions
    granularity
    n_pca
    decay
    gamma
    knn
    n_jobs

    """

    def __init__(
        self,
        scale=1.025,
        landmarks=2000,
        partitions=None,
        granularity=0.1,
        n_pca=None,
        decay=40,
        gamma=1,
        knn=5,
        n_jobs=-1,
    ):
        self.scale = scale
        self.landmarks = landmarks
        self.partitions = partitions
        self.granularity = granularity
        self.n_pca = n_pca
        self.decay = decay
        self.gamma = gamma
        self.knn = knn
        self.n_jobs = n_jobs
        self.NxTs = None
        self.Xs = None
        self.Ks = None
        self.merges = None
        self.Ps = None
        self.diff_op = None
        self.data_pca = None
        self.pca_op = None
        self.partition_clusters = None
        self.dp_pca = None
        self.epsilon = None
        self.merge_threshold = None
        self.gradient = None
        self.levels = None

        super().__init__()

    def fit(self, X):
        """Short summary.

        Parameters
        ----------
        X : type
            Description of parameter `X`.

        Returns
        -------
        type
            Description of returned object.

        """
        self.X = X
        self.hash = utils.hash_object(X)
        (
            self.NxTs,
            self.Xs,
            self.Ks,
            self.merges,
            self.Ps,
            self.diff_op,
            self.data_pca,
            self.pca_op,
            self.partition_clusters,
            self.dp_pca,
            self.epsilon,
            self.merge_threshold,
        ) = tree.build_tree(
            X,
            scale=self.scale,
            landmarks=self.landmarks,
            partitions=self.partitions,
            granularity=self.granularity,
            n_pca=self.n_pca,
            decay=self.decay,
            gamma=self.gamma,
            knn=self.knn,
            n_jobs=self.n_jobs,
        )

        self.gradient = embed.compute_gradient(self.Xs, self.merges)
        self.levels = embed.get_levels(self.gradient)

        return self.levels

    def transform(
        self,
        visualization_level=None,
        cluster_level=None,
        coarse_cluster_level=None,
        coarse_cluster=None,
        repulse=False,
    ):
        """Short summary.

        Parameters
        ----------
        X : type
            Description of parameter `X`.

        Returns
        -------
        type
            Description of returned object.

        """
        if (
            visualization_level is None
            and cluster_level is None
            and coarse_cluster_level is None
            and coarse_cluster is None
        ):
            return visualize.get_visualization(
                self.Xs, self.NxTs, self.levels[-2], self.levels[2], repulse
            )
        elif coarse_cluster_level is None and coarse_cluster is None:
            return visualize.get_visualization(
                self.Xs, self.NxTs, cluster_level, visualization_level, repulse
            )
        else:
            return embed.get_zoom_visualization(
                self.Xs,
                self.NxTs,
                visualization_level,
                cluster_level,
                coarse_cluster_level,
                coarse_cluster,
                self.n_jobs,
            )

    def build_tree(self):
        """Short summary.

        Parameters
        ----------
        X : type
            Description of parameter `X`.

        Returns
        -------
        type
            Description of returned object.

        """
        return visualize.build_condensation_tree(
            self.data_pca, self.diff_op, self.NxTs, self.merges, self.Ps
        )

    def get_tree_clusters(self, cluster_level):
        """Short summary.

        Parameters
        ----------
        X : type
            Description of parameter `X`.

        Returns
        -------
        type
            Description of returned object.

        """
        return visualize.map_clusters_to_tree(self.NxTs[cluster_level], self.NxTs)

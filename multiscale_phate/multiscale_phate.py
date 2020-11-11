from . import tree, embed, utils, visualize


class Multiscale_PHATE(object):
    """Multscale PHATE operator which performs dimensionality reduction and clustering across granularities.

    Parameters
    ----------
    scale : float, default: 1.025
        speed at which epsilon increases from iteration to
        iteration
    landmarks : int, default: 2000
        number of landmarks to compute diffusion potential
        coordinates on
    partitions : int, default: None
        number of partitions to split data into in initial
        coarse graining. Only applies ot large datasets
    granularity : float, default: .1
        Fraction of silverman bandwidth to set initial
         kernel bandwidth to
    n_pca : int, default: 100
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time.
    decay : int, default: 40
        sets decay rate of kernel tails in diffusion potential
        calculation. If None, alpha decaying kernel is not used
    gamma : float, optional, default: 1
        Informational distance constant between -1 and 1.
        `gamma=1` gives the diffusion potential log potential,
        while `gamma=0` gives a square root potential.
    knn : int, optional, default: 5
        number of nearest neighbors on which to build kernel for
        diffusion potential calculation.
    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

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
        n_jobs=1,
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
        """Builds a Diffusion Condensation tree to the data and computes
        ideal levels of granularity via gradient analysis.

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix` and `pd.DataFrame`.

        Returns
        -------
       multiscale_phate_operator : Multiscale_PHATE
        The estimator object
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

    def transform(self, visualization_level=None, cluster_level=None, coarse_cluster_level = None, coarse_cluster = None, repulse = False):
        """ Computes Multiscale PHATE embedding at a particular visualization
        and clustering granularity.
        Parameters
        ----------
        visualization_level : int, default = levels[-2]
            Resolution of Diffusion Condensation tree to embed.
        cluster_level : int, default = levels[2]
            Resolution of Diffusion Condensation tree to visualize clusters.
        coarse_cluster_level : int, default = None
            Resolution of Diffusion Condensation tree at which to identify
            clusters. Setting this variable to a level of the tree allows
            for 'zoom in' capabilities when the cluster at this resolution
            is set by 'coarse_cluster'.
        coarse_cluster : int, default = None
            Cluster in 'coarse_cluster_level' to zoom in on.
        repulse  : bool, default = False
            Allows for repulsion between points in multiscale embedding.
        Returns
        -------
        embedding : array, shape=[number of points in visualization_level, 2]
            Aggregated points embedded in a lower dimensional space using
            Multiscale PHATE
        clusters : array shape = [number of points in visualization_level]
            Cluster labels of aggregated points as found at the granularity
            of cluster_level
        sizes : array shape = [number of coarse grained samples]
            Number of points aggregated into each point as visualized at
            the granularity of visualization_level
        """
        if visualization_level is None and cluster_level is None and coarse_cluster_level is None and coarse_cluster is None:
            return visualize.get_visualization(
                self.Xs, self.NxTs, self.levels[-2], self.levels[2], repulse
            )
        elif coarse_cluster_level is None and coarse_cluster is None:
            return visualize.get_visualization(
                self.Xs, self.NxTs, cluster_level, visualization_level, repulse
            )
        else:
            return embed.get_zoom_visualization(
                self.Xs, self.NxTs, visualization_level, cluster_level, coarse_cluster_level, coarse_cluster,
                self.n_jobs
            )

    def build_tree(self):
        """Computes and returns a tree from the Diffusion Condensation process.

        Parameters
        ----------
        None

        Returns
        -------
        embedding : array, shape=[number of aggregated points
        across granularities, 3]
            Embedding stacked 2D condensed representations of the Diffusion
            Condensation process as computed on X
        """
        return visualize.build_condensation_tree(
            self.data_pca, self.diff_op, self.NxTs, self.merges, self.Ps
        )

    def fit_transform(self, X):
        """Builds a Diffusion Condensation tree to the data and computes
        ideal levels of granularity via gradient analysis before
        computing a Multiscale PHATE embedding at a ideal visualization
        and clustering granularities.

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix` and `pd.DataFrame`.
        Returns
        -------
        embedding : array, shape=[number of points in visualization_level, 2]
            Aggregated points embedded in a lower dimensional space using
            Multiscale PHATE
        clusters : array shape = [number of points in visualization_level]
            Cluster labels of aggregated points as found at the granularity
            of cluster_level
        sizes : array shape = [number of coarse grained samples]
            Number of points aggregated into each point as visualized at
            the granularity of visualization_level
        """
        self.fit(X)
        return self.transform(X)

    def get_tree_clusters(self, cluster_level):
        """Colors Diffusion Condensation tree by a granularity of clusters.

        Parameters
        ----------
        cluster_level : int
            Resolution of Diffusion Condensation tree to visualize clusters.

        Returns
        -------
        clusters_tree : array shape = array, shape=[number of aggregated points
        across granularities, 1]
            Cluster labels of each point in computed diffusion condensation tree
            as dictated by a granularity of the tree

        """
        return visualize.map_clusters_to_tree(self.NxTs[cluster_level], self.NxTs)





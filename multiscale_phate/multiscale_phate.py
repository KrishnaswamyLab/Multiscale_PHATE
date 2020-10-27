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
        landmarks=1000,
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
        return self

    def transform(self, X):
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
        if utils.hash_object(X) == self.hash:
            NxTs = self.NxTs
            Xs = self.Xs
            Ks = self.Ks
            merges = self.merges
            Ps = self.Ps
            data_pca = self.data_pca
        else:
            NxTs, Xs, Ks, merges, Ps, data_pca = tree.online_update_tree(
                self.X,
                X,
                self.data_pca,
                self.pca_op,
                self.partition_clusters,
                self.diff_op,
                self.dp_pca,
                self.Xs,
                self.NxTs,
                self.Ks,
                self.merges,
                self.Ps,
                scale=self.scale,
                n_jobs=self.n_jobs,
            )

        hp_embedding, cluster_viz, sizes_viz = visualize.build_visualization(
            Xs, NxTs, merges
        )

        vis_tree = visualize.build_condensation_tree(
            data_pca, self.diff_op, NxTs, merges, Ps
        )

        return hp_embedding, cluster_viz, sizes_viz, vis_tree

    def fit_transform(self, X):
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
        self.fit(X)
        return self.transform(X)

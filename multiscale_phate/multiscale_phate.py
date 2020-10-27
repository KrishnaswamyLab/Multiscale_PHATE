from . import tree, embed, utils, visualize


class Multiscale_PHATE:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X):
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
        ) = tree.build_tree(X, n_jobs=10)
        return self

    def transform(self, X):
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
                self.Ks,
                self.merges,
                self.Ps,
                1.025,
                10,
            )

        hp_embedding, cluster_viz, sizes_viz = visualize.build_visualization(
            Xs, NxTs, merges
        )

        vis_tree = visualize.build_condensation_tree(
            data_pca, self.diff_op, NxTs, merges, Ps
        )

        return hp_embedding, cluster_viz, sizes_viz, vis_tree

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

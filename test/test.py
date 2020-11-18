import numpy as np
import multiscale_phate
import warnings
import parameterized

warnings.simplefilter("error")


@parameterized.parameterized([(None, None), (100, None), (None, 50), (100, 50)])
def test(partitions, landmarks):
    X = np.random.normal(0, 1, (200, 200))
    mp_op = multiscale_phate.Multiscale_PHATE(
        partitions=partitions,
        landmarks=landmarks,  # n_pca=20
    )
    hp_embedding, cluster_viz, sizes_viz = mp_op.fit_transform(X)
    return
    assert hp_embedding.shape[0] <= X.shape[0], (X.shape, hp_embedding.shape)
    assert hp_embedding.shape[1] == 2, (X.shape, hp_embedding.shape)
    assert cluster_viz.shape == (hp_embedding.shape[0],), (
        X.shape,
        hp_embedding.shape,
        cluster_viz.shape,
    )
    assert sizes_viz.shape == (hp_embedding.shape[0],), (
        X.shape,
        hp_embedding.shape,
        sizes_viz.shape,
    )
    # assert tree.shape[1] == 3, (X.shape, tree.shape)

    Y = np.random.normal(0.5, 1, (200, 200))
    hp_embedding, cluster_viz, sizes_viz = mp_op.fit_transform(Y)
    assert hp_embedding.shape[0] <= X.shape[0] + Y.shape[0], (
        X.shape,
        Y.shape,
        hp_embedding.shape,
    )
    assert hp_embedding.shape[1] == 2, (X.shape, Y.shape, hp_embedding.shape)
    assert cluster_viz.shape == (hp_embedding.shape[0],), (
        X.shape,
        Y.shape,
        hp_embedding.shape,
        cluster_viz.shape,
    )
    assert sizes_viz.shape == (hp_embedding.shape[0],), (
        X.shape,
        Y.shape,
        hp_embedding.shape,
        sizes_viz.shape,
    )

    tree = mp_op.build_tree()
    tree_clusters = mp_op.get_tree_clusters(mp_op.levels[-2])

    assert tree.shape[0] == len(tree_clusters)


# assert tree.shape[1] == 3, (X.shape, Y.shape, tree.shape)


def test_random_seed():
    X = np.random.normal(0, 1, (200, 200))

    mp_op = multiscale_phate.Multiscale_PHATE(
        partitions=100,
        landmarks=50,
        random_state=42,  # n_pca=20
    )
    hp_embedding, _, _ = mp_op.fit_transform(X)
    hp_embedding2, _, _ = mp_op.fit_transform(X)
    np.testing.assert_equal(hp_embedding, hp_embedding2)

    mp_op = multiscale_phate.Multiscale_PHATE(partitions=100, landmarks=50)
    hp_embedding, _, _ = mp_op.fit_transform(X)
    hp_embedding2, _, _ = mp_op.fit_transform(X)
    # np.testing.assert_all_close(hp_embedding, hp_embedding2)

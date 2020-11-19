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
    if hp_embedding.shape[0] == hp_embedding2.shape[0]:
        assert not np.all(hp_embedding == hp_embedding2)


@parameterized.parameterized(
    [
        # n_pca is None -> min(N, features)
        (100, 50, None, 50),
        (50, 100, None, 50),
        # n_pca < min(N, features) -> n_pca
        (100, 50, 25, 25),
        # n_pca > 100 -> 100
        (200, 150, 200, 100),
        (200, 150, 125, 100),
        # n_pca > min(N, features) -> min(N, features)
        (100, 50, 75, 50),
        (50, 100, 75, 50),
        (100, 50, 125, 50),
        (50, 100, 125, 50),
    ]
)
def test_compression_features_pca(N, features, n_pca, expected):
    partitions = None
    output, _ = multiscale_phate.compress.get_compression_features(
        N, features, n_pca, partitions
    )
    assert output == expected


@parameterized.parameterized(
    [
        # TODO: is this desired behavior? seems pathological
        # partitions is None -> None
        (100, None, None),
        # partitions > N -> None
        (100, 101, None),
        (200000, 200001, None),
        # partitions > 50000 -> 50000
        (110000, 50001, 50000),
        # N > 100000 -> 20000
        (110000, None, 20000),
        (110000, 100, 20000),
        (110000, 50000, 20000),
        (110000, 110001, 20000),
    ]
)
def test_compression_features_partitions(N, partitions, expected):
    n_pca = None
    features = 50
    _, output = multiscale_phate.compress.get_compression_features(
        N, features, n_pca, partitions
    )
    assert output == expected

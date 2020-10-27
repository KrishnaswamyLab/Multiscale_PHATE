import numpy as np
import multiscale_phate
import warnings

warnings.simplefilter("error")
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Pre-fit PHATE should not be used to transform a new data matrix. "
    "Please fit PHATE to the new data by running 'fit' with the new data.",
)


def test():
    X = np.random.normal(0, 1, (100, 200))
    mp_op = multiscale_phate.Multiscale_PHATE()
    hp_embedding, cluster_viz, sizes_viz, tree = mp_op.fit_transform(X)
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
    assert tree.shape[1] == 3, (X.shape, tree.shape)

    Y = np.random.normal(0.5, 1, (50, 200))
    hp_embedding, cluster_viz, sizes_viz, tree = mp_op.transform(Y)
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
    assert tree.shape[1] == 3, (X.shape, Y.shape, tree.shape)

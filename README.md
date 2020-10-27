Multiscale_PHATE
================

[![Latest PyPi version](https://img.shields.io/pypi/v/multiscale_phate.svg)](https://pypi.org/project/multiscale_phate/)
[![Travis CI Build](https://api.travis-ci.com/KrishnaswamyLab/Multiscale_PHATE.svg?branch=master)](https://travis-ci.com/KrishnaswamyLab/Multiscale_PHATE)
[![Coverage Status](https://coveralls.io/repos/github/KrishnaswamyLab/Multiscale_PHATE/badge.svg?branch=master)](https://coveralls.io/github/KrishnaswamyLab/Multiscale_PHATE?branch=master)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![GitHub stars](https://img.shields.io/github/stars/KrishnaswamyLab/Multiscale_PHATE.svg?style=social&label=Stars)](https://github.com/KrishnaswamyLab/Multiscale_PHATE/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is a short description of the package.

Installation
------------

Multiscale_PHATE is available on `pip`. Install by running the following in a terminal:

```
pip install --user git+https://github.com/KrishnaswamyLab/Multiscale_PHATE
```

Quick start
-----------

```
import numpy as np
X = np.random.normal(0, 1, (100, 10))

import multiscale_phate
mp_op = multiscale_phate.Multiscale_PHATE()
hp_embedding, cluster_viz, sizes_viz, tree = mp_op.fit_transform(X)

# Plot optimal visualization
scprep.plot.scatter2d(hp_embedding, s = sizes_viz, c = cluster_viz,
                      fontsize=16, ticks=False,label_prefix="Multiscale-PHATE", figsize=(16,12))

# Plot condensation tree
scprep.plot.scatter3d(tree, c=tree[:,2],fontsize=16, ticks=False, label_prefix="C-PHATE", figsize=(16,12), s=20)

# Embed online data
Y = np.random.normal(0.5, 1, (50, 10))
hp_embedding, cluster_viz, sizes_viz, tree = mp_op.transform(Y)
```

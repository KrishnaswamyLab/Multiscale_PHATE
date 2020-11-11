Multiscale PHATE
================

[![Latest PyPi version](https://img.shields.io/pypi/v/multiscale_phate.svg)](https://pypi.org/project/multiscale_phate/)
[![Travis CI Build](https://api.travis-ci.com/KrishnaswamyLab/Multiscale_PHATE.svg?branch=master)](https://travis-ci.com/KrishnaswamyLab/Multiscale_PHATE)
[![Coverage Status](https://coveralls.io/repos/github/KrishnaswamyLab/Multiscale_PHATE/badge.svg?branch=master)](https://coveralls.io/github/KrishnaswamyLab/Multiscale_PHATE?branch=master)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![GitHub stars](https://img.shields.io/github/stars/KrishnaswamyLab/Multiscale_PHATE.svg?style=social&label=Stars)](https://github.com/KrishnaswamyLab/Multiscale_PHATE/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Multiscale PHATE is a python package for multiresolution analysis of extremely high dimensional data. For an in-depth explanation of the algorithm and applications, please read our manuscript on BioRxiv:

LINK TO ARTICLE

The biomedical community is producing increasingly high dimensional datasets integrated from hundreds of patient samples that current computational techniques are unable to explore. Current tools for dimensionality reduction, such as tSNE, UMAP, and PCA, and clustering, such as Louvain and Leiden, only show a single salient level of granularity in biomedical data. When applied to cellular datasets currently being produced, these techniques are able to visualize and cluster major cell types such as B cells, T cells and myeloid cells. Differences between patient disease states, however, may not be found at the granularity of cell type alone. In fact, appreciation of a finer resolution the manifold would reveal subsets that may be predictive of outcome. This phenomenon is found across biomedical data science, as the cellular state space is known to form a collection of sub-manifolds that disease status can differentially affect.

The goal of Multiscale PHATE is to learn and visualize abstract cellular features and groupings of the data at all levels of granularity in an efficient manner to identify meaningful resolutions. Our approach learns a tree of data granularities which can be cut at coarse levels for high level summarizations of data as well as at fine levels for detailed representations on subsets. Our algorithm is based on a dynamic process we have developed called diffusion condensation, that computes a manifold-intrinsic diffusion space on the original data before slowly condensing data points towards local centers of gravity to form natural, data-driven groupings across multiple granularities.  While this may sound computationally inefficient, we show that we are able to perform these calculations as well as visualize and cluster the data significantly faster than “single-scale” visualization techniques like tSNE, UMAP or PHATE, allowing the analysis of millions of cells within minutes.  When combined with other computational algorithms for high dimensional data analysis, such as MELD, DREMI and TrajcetoryNet, Multiscale PHATE is able to provide deep and detailed insights in biological processes.

Installation
------------

Multiscale PHATE is available on `pip`. Install by running the following in a terminal:

```
pip install --user git+https://github.com/KrishnaswamyLab/Multiscale_PHATE
```

Quick Start
-----------

```
import multiscale_phate
mp_op = multiscale_phate.Multiscale_PHATE()
mp_embedding, mp_clusters, mp_sizes, tree = mp_op.fit_transform(X)

# Plot optimal visualization
scprep.plot.scatter2d(mp_embedding, s = mp_sizes, c = mp_clusters,
                      fontsize=16, ticks=False,label_prefix="Multiscale PHATE", figsize=(16,12))
```

Guided Tutorial
-----------

TO DO: Put Tutorial link here
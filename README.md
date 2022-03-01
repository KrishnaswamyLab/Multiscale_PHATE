Multiscale PHATE
================

[![Latest PyPi version](https://img.shields.io/pypi/v/multiscale_phate.svg)](https://pypi.org/project/multiscale_phate/)
[![Travis CI Build](https://api.travis-ci.com/KrishnaswamyLab/Multiscale_PHATE.svg?branch=master)](https://travis-ci.com/KrishnaswamyLab/Multiscale_PHATE)
[![Coverage Status](https://coveralls.io/repos/github/KrishnaswamyLab/Multiscale_PHATE/badge.svg?branch=master)](https://coveralls.io/github/KrishnaswamyLab/Multiscale_PHATE?branch=master)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![GitHub stars](https://img.shields.io/github/stars/KrishnaswamyLab/Multiscale_PHATE.svg?style=social&label=Stars)](https://github.com/KrishnaswamyLab/Multiscale_PHATE/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Multiscale PHATE is a python package for multiresolution analysis of high dimensional data. For an in-depth explanation of the algorithm and applications, please read our manuscript on [Nature Biotechnology](https://www.nature.com/articles/s41587-021-01186-x).

The biomedical community is producing increasingly high dimensional datasets integrated from hundreds of patient samples that current computational techniques are unable to explore across granularities. To visualize, cluster and analyze massive datasets across granularities, we created Multiscale PHATE. The goal of Multiscale PHATE is to learn and visualize abstract cellular features and groupings of the data at all levels of granularity in an efficient manner to identify meaningful biological relationships and mechanisms. Our approach learns a tree of data granularities which can be cut at coarse levels for high level summarizations of data as well as at fine levels for detailed representations on subsets.


# Overview of Algorithm:

![alt text](https://github.com/KrishnaswamyLab/Multiscale_PHATE/blob/main/Images/Algorithm%20Overview.png)

Our algorithm integrates dimensionality reduction technique [PHATE](https://www.nature.com/articles/s41587-019-0336-3) with multigranular analysis tool [diffusion condensation](https://ieeexplore.ieee.org/document/9006013). First the non-linear diffusion manifold is calculated using PHATE. Then diffusion condensation takes this manifold-intrinsic diffusion space and slowly condensing data points towards local centers of gravity to form natural, data-driven groupings across multiple granularities. These granularities can then be viewed.


![alt text](https://github.com/KrishnaswamyLab/Multiscale_PHATE/blob/main/Images/Gradient%20Analysis.png)
Using gradient analysis, which looks at shifts in data density during successive iterations of the diffusion condensation process, we can identify stable resolutions of the hierarchical tree for downstream analysis. With this stability information, we can cut the hierarchical tree at multiple resolutions to produce visualizations and clusters across granularities for downstream analysis.

![alt text](https://github.com/KrishnaswamyLab/Multiscale_PHATE/blob/main/Images/Zoom%20in.png)
By identifying multiple resolutions, Multiscale PHATE enables users to interact with their data and zoom in on cellular subsets of interest to reveal increasingly granular information about cell types and subtypes.

While this may sound computationally inefficient, we show that we are able to perform these calculations as well as visualize and cluster the data significantly faster than “single-scale” visualization techniques like tSNE, UMAP or PHATE, allowing the analysis of millions of cells within minutes.  When combined with other computational algorithms for high dimensional data analysis, such as MELD and DREMI, Multiscale PHATE is able to provide deep and detailed insights in biological processes.

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
mp_embedding, mp_clusters, mp_sizes = mp_op.fit_transform(X)

# Plot optimal visualization
scprep.plot.scatter2d(mp_embedding, s = mp_sizes, c = mp_clusters,
                      fontsize=16, ticks=False,label_prefix="Multiscale PHATE", figsize=(16,12))
```

Guided Tutorial
-----------

For more details on using Multiscale PHATE, see our [guided tutorial](https://github.com/KrishnaswamyLab/Multiscale_PHATE/blob/main/Tutorial/10X_pbmc.ipynb) using 10X's public PBMC4k dataset.

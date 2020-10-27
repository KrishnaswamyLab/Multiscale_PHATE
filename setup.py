import os
import sys
from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "scipy",
    "graphtools",
    "phate",
    "scikit-learn",
    "tasklogger",
    "joblib",
]

test_requires = ["nose2", "numpy", "coverage", "coveralls", "parameterized", "black"]

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >=3.6 required.")

version_py = os.path.join(os.path.dirname(__file__), "multiscale_phate", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

readme = open("README.md").read()

setup(
    name="multiscale_phate",
    version=version,
    description="multiscale_phate",
    author="Manik Kuchroo & Scott Gigante, Yale University",
    author_email="manik.kuchroo@yale.edu",
    packages=find_packages(),
    include_package_data=True,
    license="GNU General Public License Version 3",
    install_requires=install_requires,
    python_requires=">=3.6",
    extras_require={"test": test_requires},
    test_suite="nose2.collector.collector",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/KrishnaswamyLab/Multiscale_PHATE",
    download_url="https://github.com/KrishnaswamyLab/Multiscale_PHATE/archive/v{}.tar.gz".format(
        version
    ),
    keywords=[
        "big-data",
        "computational-biology",
        "dimensionality-reduction",
        "visualization",
        "embedding",
        "manifold-learning",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)

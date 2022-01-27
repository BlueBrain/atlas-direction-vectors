#!/usr/bin/env python

import imp

from setuptools import find_packages, setup

VERSION = imp.load_source("", "atlas_direction_vectors/version.py").__version__

setup(
    name="atlas-direction-vectors",
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="Commands to compute direction vectors in volumetric brain regions. "
    "Direction vectors are 3D unit vectors associated to voxels of a brain region. "
    "They represent the directions of the fiber tracts.",
    url="https://bbpgitlab.epfl.ch/nse/atlas-direction-vectors",
    download_url="git@bbpgitlab.epfl.ch:nse/atlas-direction-vectors.git",
    license="BBP-internal-confidential",
    python_requires=">=3.6.0",
    install_requires=[
        "atlas-commons>=0.1.2",
        "click>=7.0",
        "nptyping>=1.0.1",
        "numpy>=1.15.0",
        "numpy-quaternion[numba]>=2021.11.4.15.26.3",
        "scipy>=1.4.1",
        "voxcell>=3.0.0",
    ],
    extras_require={
        "tests": ["pytest>=4.4.0", "mock>=2.0.0"],
    },
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": ["atlas-direction-vectors=atlas_direction_vectors.app.cli:cli"]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)

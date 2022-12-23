#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.rst") as f:
    README = f.read()

setup(
    name="atlas-direction-vectors",
    author="Blue Brain Project, EPFL",
    description=(
        "Commands to compute direction vectors in volumetric brain regions. "
        "Direction vectors are 3D unit vectors associated to voxels of a brain region. "
        "They represent the directions of the fiber tracts."
    ),
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/BlueBrain/atlas-direction-vectors",
    download_url="https://github.com/BlueBrain/atlas-direction-vectors",
    license="Apache-2",
    python_requires=">=3.7.0",
    install_requires=[
        "atlas-commons>=0.1.4",
        "click>=7.0",
        "numpy>=1.15.0",
        "scipy>=1.4.1",
        "voxcell>=3.0.0",
    ],
    extras_require={
        "tests": [
            "pytest>=4.4.0",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": ["atlas-direction-vectors=atlas_direction_vectors.app.cli:cli"]
    },
    use_scm_version={
        "local_scheme": "no-local-version",
    },
    setup_requires=[
        "setuptools_scm",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

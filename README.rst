Overview
=========

This project contains the commands which create direction vectors for several brain regions including
the cerebellum, the isocortex, and the thalamus of the AIBS P56 mouse brain.

Direction vectors are 3D unit vectors associated to voxels of a brain region.
They represent the directions of the fiber tracts and their streamlines are assumed
to cross transversely layers in laminar brain regions.

Direction vectors are used in placement-algorithm to set cells orientations.

Direction vectors are also used to compute placement hints (see the placement_hints module)
and split layer 2/3 of the AIBS mouse isocortex.

After installation, you can display the available command lines with the following ``bash`` command:

.. code-block:: bash

    atlas-direction-vectors --help

Installation
============

.. code-block:: bash

    git clone git@bbpgitlab.epfl.ch:nse/atlas-direction-vectors.git
    cd atlas-direction-vectors
    pip install -e .


Instructions for developers
===========================

Run the following commands before submitting your code for review:

.. code-block:: bash

    cd atlas-direction-vectors
    isort -l 100 --profile black atlas_direction_vectors tests setup.py
    black -l 100 atlas_direction_vectors tests setup.py

These formatting operations will help you pass the linting check `testenv:lint` defined in
`tox.ini`.
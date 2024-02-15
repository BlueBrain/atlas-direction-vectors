"""
Function computing the direction vectors of the mouse olfactory bulb
"""
from __future__ import annotations

import logging

import numpy as np
from atlas_commons.typing import NDArray
from atlas_commons.utils import get_region_mask
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_direction_vectors.algorithms import layer_based_direction_vectors
from atlas_direction_vectors.utils import warn_on_nan_vectors

L = logging.getLogger(__name__)
logging.captureWarnings(True)


def compute_direction_vectors(region_map: RegionMap, annotation: VoxelData) -> NDArray[np.float32]:
    """Returns the direction vectors computed

    Arguments:
        region_map: RegionMap object with the hierarchy data
        annotation: VoxelData object containing the isocortex or a superset.

    Returns:
        numpy.ndarray of shape (annotation.shape, 3) holding a 3D unit vector field.
    """

    metadata = {
        "region": {
            "name": "Main olfactory bulb",
            "query": "@^\\bMOB|lfbs\\b$",
            "attribute": "acronym",
            "with_descendants": True,
        },
        "layers": {
            "names": [
                "glomerular",
                "outer_plexiform",
                "mitral",
                "inner_plexiform",
                "granule",
            ],
            "queries": [
                "MOBgl",
                "MOBopl",
                "MOBmi",
                "MOBipl",
                "MOBgr",
            ],
            "attribute": "acronym",
            "with_descendants": True,
        },
    }
    region_to_weight = {
        "MOBgl": 5,
        "MOBopl": 4,
        "MOBmi": 3,
        "MOBipl": 2,
        "MOBgr": 1,
        "outside_of_brain": 0,
    }

    direction_vectors = layer_based_direction_vectors.compute_layered_region_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        metadata=metadata,
        region_to_weight=region_to_weight,
        shading_width=4,
        expansion_width=8,
        has_hemispheres=True,
    )

    warn_on_nan_vectors(
        direction_vectors, get_region_mask("MOB", annotation.raw, region_map), "MOB"
    )

    return direction_vectors

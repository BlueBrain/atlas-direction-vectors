"""Function computing the direction vectors of the AIBS mouse cerebellum

The algorithm creates a scalar field with low values in surfaces where fiber tracts are incoming
and high values where fiber tracts are outgoing. The direction vectors are given by the gradient
of this scalar field.

Note: At the moment, direction vectors are generated only for the following cerebellum subregions:
    - the flocculus
    - the lingula
"""
from typing import TYPE_CHECKING

import numpy as np
from atlas_commons.typing import FloatArray

from atlas_direction_vectors.algorithms.layer_based_direction_vectors import (
    compute_layered_region_direction_vectors,
)

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import RegionMap, VoxelData  # type: ignore


def compute_direction_vectors(region_map: "RegionMap", annotation: "VoxelData") -> FloatArray:
    """
    Computes cerebellum's direction vectors as the normalized gradient of a custom scalar field.

    The computations are restricted to the flocculus and the lingula subregions.

    The output direction vector field is computed as the normalized gradient
    of a custom scalar field. This scalar field resembles a distance field in
    the neighborhood of the molecular layer.

    Afterwards, a Gaussian filter is applied and the normalized gradient of the
    blurred scalar field is returned.

    Note: For now, direction vectors are only computed for the flocculus and lingula subregions.
        A voxel lying outside these two regions will be assigned a 3D vector
        with np.nan coordinates.

    Arguments:
        region_map: hierarchy data structure of the AIBS atlas
        annotation: integer array of shape (W, H, D) holding the annotation of the whole mouse
         brain.

    Returns:
        numpy.ndarray of shape (annotation.shape, 3) holding a 3D unit vector field.
    """
    flocculus_direction_vectors = _flocculus_direction_vectors(region_map, annotation)
    lingula_direction_vectors = _lingula_direction_vectors(region_map, annotation)

    # Assembles flocculus and lingula direction vectors.
    direction_vectors = flocculus_direction_vectors
    lingula_mask = np.logical_not(np.isnan(lingula_direction_vectors))
    direction_vectors[lingula_mask] = lingula_direction_vectors[lingula_mask]

    return direction_vectors


def _flocculus_direction_vectors(region_map: "RegionMap", annotation: "VoxelData") -> FloatArray:
    """Returns the directin vectors for the flocculus subregions

    name: cerebellum related fiber tracts, acronym: cbf,  identifier = 960
    name: Flocculus, granular layer, acronym: FLgr, identifier: 10690
    name: Flocculus, purkinje layer, acronym: FLpu, identifier: 10691
    name: Flocculus molecular layer, acronym: FLmo, identifier: 10692
    """
    metadata = {
        "region": {
            "name": "Extended Flocculus",
            "query": r"@^\bFL|cbf\b$",
            "attribute": "acronym",
            "with_descendants": True,
        },
        "layers": {
            "names": [
                "Flocculus, molecular layer",
                "Flocculus, Purkinje layer",
                "Flocculus, granular layer",
                "cerebellum related fiber tracts",
            ],
            "queries": ["FLmo", "FLpu", "FLgr", "cbf"],
            "attribute": "acronym",
            "with_descendants": True,
        },
    }
    region_to_weight = {
        "cbf": -5,
        "FLgr": -1,
        "FLpu": 0,
        "FLmo": 1,
        "outside_of_brain": 3,
    }
    return compute_layered_region_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        metadata=metadata,
        region_to_weight=region_to_weight,
        shading_width=4,
        expansion_width=8,
        has_hemispheres=False,
    )


def _lingula_direction_vectors(region_map: "RegionMap", annotation: "VoxelData") -> FloatArray:
    """Returns direction vectors for the lingula subregions

    name: cerebellum related fiber tracts, acronym: cbf,  identifier = 960
    name: Lingula molecular layer, acronym: LINGmo, identifier: 10707
    name: Lingula, purkinje layer, acronym: LINGpu, identifier: 10706
    name: Lingula, granular layer, acronym: LINGgr, identifier: 10705
    """
    metadata = {
        "region": {
            "name": "Extended Lingula",
            "query": r"@^\bLING|cbf\b$",
            "attribute": "acronym",
            "with_descendants": True,
        },
        "layers": {
            "names": [
                "Lingula (I), molecular layer",
                "Lingula (I), Purkinje layer",
                "Lingula (I), granular layer",
                "cerebellum related fiber tracts",
            ],
            "queries": ["LINGmo", "LINGpu", "LINGgr", "cbf"],
            "attribute": "acronym",
            "with_descendants": True,
        },
    }
    region_to_weight = {"cbf": -5, "LINGgr": -1, "LINGpu": 0, "LINGmo": 1, "outside_of_brain": 3}
    return compute_layered_region_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        metadata=metadata,
        region_to_weight=region_to_weight,
        shading_width=4,
        expansion_width=8,
        has_hemispheres=False,
    )

"""Function computing the direction vectors of the AIBS mouse cerebellar cortex

The algorithm creates a scalar field with low values in surfaces where fiber tracts are incoming
and high values where fiber tracts are outgoing. The direction vectors are given by the gradient
of this scalar field.
"""

import logging
from functools import partial
from typing import TYPE_CHECKING, Optional

import numpy as np
from atlas_commons.typing import FloatArray
from joblib import Parallel, delayed

from atlas_direction_vectors.algorithms.layer_based_direction_vectors import (
    compute_layered_region_direction_vectors,
)

if TYPE_CHECKING:  # pragma: no cover
    from voxcell import RegionMap, VoxelData  # type: ignore

L = logging.getLogger(__name__)


def compute_direction_vectors(region_map: "RegionMap", annotation: "VoxelData") -> FloatArray:
    """
    Computes cerebellum's direction vectors as the normalized gradient of a custom scalar field.

    The output direction vector field is computed as the normalized gradient
    of a custom scalar field. This scalar field resembles a distance field in
    the neighborhood of the molecular layer.

    Afterwards, a Gaussian filter is applied and the normalized gradient of the
    blurred scalar field is returned.

    Arguments:
        region_map: hierarchy data structure of the AIBS atlas
        annotation: integer array of shape (W, H, D) holding the annotation of the whole mouse
         brain.

    Returns:
        numpy.ndarray of shape (annotation.shape, 3) holding a 3D unit vector field.
    """
    direction_vectors = np.full(annotation.raw.shape + (3,), np.nan, dtype=np.float32)
    subregion_ids = []
    for subregion_id in region_map.find("Cerebellar cortex", "name", with_descendants=True):
        parent_id = region_map.get(subregion_id, "parent_structure_id")
        if (
            subregion_id in annotation.raw
            and region_map.is_leaf_id(subregion_id)
            and parent_id not in subregion_ids
        ):
            subregion_ids.append(parent_id)
    L.info(
        "Computing direction vectors for regions %s.",
        [region_map.get(parent_id, "name") for parent_id in subregion_ids],
    )

    f = partial(cereb_subregion_direction_vectors, region_map=region_map, annotation=annotation)
    with Parallel(n_jobs=-1, verbose=10) as parallel:
        for subregion_direction_vectors in parallel(
            delayed(f)(parent_id) for parent_id in subregion_ids
        ):
            # Assembles subregion direction vectors.
            subregion_mask = np.logical_not(np.isnan(subregion_direction_vectors))
            direction_vectors[subregion_mask] = subregion_direction_vectors[subregion_mask]

    return direction_vectors


def cereb_subregion_direction_vectors(
    region_id: int, region_map: "RegionMap", annotation: "VoxelData", weights: Optional[dict] = None
) -> FloatArray:
    """Returns the direction vectors for a cerebellar cortex subregion
    Arguments:
        region_id: id of the cerebellar cortex subregion of interest
        region_map: hierarchy data structure of the AIBS atlas
        annotation: integer array of shape (W, H, D) holding the annotation of the whole mouse
         brain.
        weights: dictionary linking the cerebellar cortex regions' acronym ("mo", "pu", "gr", "cbf",
         "outside_of_brain") to their weight in the custom scalar field (default is respectively:
          1, 0, -1, -5, 3).
    Returns:
        numpy.ndarray of shape (annotation.shape, 3) holding a 3D unit vector field.
    """
    if weights is None:
        weights = {}
    acronym = region_map.get(region_id, "acronym")
    name = region_map.get(region_id, "name")

    # first elem is molecular layer, last element is the fibers
    metadata = {
        "region": {
            "name": "Cerebellar cortex subreg",
            "query": rf"@^\b{acronym}|cbf\b$",
            "attribute": "acronym",
            "with_descendants": True,
        },
        "layers": {
            "names": [
                name + ", molecular layer",
                name + ", granular layer",
                "cerebellum related fiber tracts",
            ],
            "queries": [acronym + "mo", acronym + "gr", "cbf"],
            "attribute": "acronym",
            "with_descendants": True,
        },
    }
    region_to_weight = {
        acronym + "mo": weights.get("mo", 1.0),
        acronym + "gr": weights.get("gr", -1.0),
        "cbf": weights.get("cbf", -5.0),
        "outside_of_brain": weights.get("outside_of_brain", 3),
    }

    return compute_layered_region_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        metadata=metadata,
        region_to_weight=region_to_weight,
        shading_width=4,
        expansion_width=8,
        has_hemispheres=True,
    )

"""
Function computing the direction vectors of the mouse isocortex
"""
from __future__ import annotations

import logging
import re
from functools import partial
from typing import Callable, Dict, List

import numpy as np
from atlas_commons.typing import AnnotationT, NDArray
from atlas_commons.utils import get_region_mask
from voxcell import RegionMap, VoxelData  # type: ignore
from voxcell.math_utils import minimum_aabb  # pylint: disable=ungrouped-imports

from atlas_direction_vectors.algorithms import layer_based_direction_vectors
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError
from atlas_direction_vectors.utils import warn_on_nan_vectors

L = logging.getLogger(__name__)
logging.captureWarnings(True)
# The endings of names and acronyms in the 6 layers of the AIBS mouse isocortex are:
#  * 1, 2, 3, 4, 5
#  * 2/3
#  * 6a, 6b
# Reference: AIBS 1.json as of 2020/04.
# We search for these endings with the following regular expression:
LAYER_ENDINGS = "^([a-zA-Z]*-?[a-zA-Z]+)(?:[1-5]|2/3|6[ab])$"

# pylint: disable=fixme
# TODO: get layered subregions, excluding unrepresented
# including non-leaf represented from the hierarchy
# pylint: enable=fixme


def get_isocortical_regions(annotation: AnnotationT, region_map: RegionMap) -> List[str]:
    """
    Get the acronyms of all isocortical regions present in `annotation`.

    Args:
        annotation: 3D array of region identifiers containing the isocortex ids.
        region_map: a RegionMap to navigate the brain regions hierarchy.

    Returns:
        A list containing the isocortical region acronyms.

    Note: The output list may vary from one annotation file to the other.
    For the Mouse ccfv2 annotation with a resolution of 25um, 40 acronyms
    are returned. For the Mouse ccfv3 annotation of the same resolution,
    43 acronyms are returned.
    """
    isocortex_mask = get_region_mask("Isocortex", annotation, region_map)
    ids = np.unique(annotation[isocortex_mask])
    acronyms = set()
    for id_ in ids:
        acronym = region_map.get(id_, "acronym")
        search = re.search(LAYER_ENDINGS, acronym)
        if search is not None:
            acronym = search.group(1)
            acronyms |= {acronym}

    return sorted(list(acronyms))


def _direction_vectors_per_region(
    region_map: RegionMap,
    annotation: VoxelData,
    algorithm: str = "regiodesics",
) -> NDArray[np.float32]:
    """
    Compute the mouse isocortex direction vectors.

    Arguments:
        region_map: RegionMap object.
        annotation: VoxelData object containing the isocortex or a superset.
        algorithm: Algorithm to use for the computation of the direction vectors.
            By default `regiodesics` is used

    Returns:
        Vector field of 3D unit vectors over the isocortex volume with the same shape
        as the input one. Voxels outside the Isocortex have np.nan coordinates.
    """
    direction_vectors = np.full(annotation.shape + (3,), np.nan, dtype=np.float32)
    # Get the highest-level regions of the isocortex: ACAd, ACAv, AId, AIp, AIv, ...
    # In the AIBS mouse ccfv3 annotation, there are 43 isocortical regions.
    regions = get_isocortical_regions(annotation.raw, region_map)

    for region in regions:
        L.info("Computing direction vectors for region %s", region)
        region_mask = get_region_mask(region, annotation.raw, region_map)
        # pylint: disable=not-an-iterable
        aabb_slice = tuple(
            slice(bottom, top + 1) for (bottom, top) in np.array(minimum_aabb(region_mask)).T
        )
        voxel_data = annotation.with_data(annotation.raw[aabb_slice])
        try:
            region_direction_vectors = layer_based_direction_vectors.compute_direction_vectors(
                region_map,
                voxel_data,
                {
                    # Note: layer 6b matches with the bottom boundary of the isocortex.
                    # Layer 6a doesn't.
                    "source": [("acronym", "@.*6[b]$")],
                    "inside": [("acronym", region)],
                    "target": [("acronym", "@.*1$")],
                },
                algorithm=algorithm,
                hemisphere_opposite_option=(
                    layer_based_direction_vectors.HemisphereOppositeOption.INCLUDE_AS_TARGET
                ),
            )
        except AtlasDirectionVectorsError as error:
            L.warning(error)
            L.warning(
                "Direction vectors computation failed for region %s: direction vectors are "
                "set to (NaN, NaN, NaN) in this region.",
                region,
            )
            continue

        region_mask = region_mask[aabb_slice]
        direction_vectors[aabb_slice][region_mask] = region_direction_vectors[region_mask]
        del region_direction_vectors

    return direction_vectors


def _shading_gradient(region_map: RegionMap, annotation: VoxelData) -> NDArray[np.float32]:
    """
    Computes isocortex's direction vectors as the normalized gradient of a custom scalar field.

    The output direction vector field is computed as the normalized gradient
    of a custom scalar field. This scalar field resembles a distance field in
    the neighborhood of the layer 1.

    Afterwards, a Gaussian filter is applied and the normalized gradient of the
    blurred scalar field is returned.

    Arguments:
        annotation_raw: integer array of shape (W, H, D) holding the annotation of the whole mouse
         brain.
        region_map: RegionMap object with the hierarchy data

    Returns:
        numpy.ndarray of shape (annotation.shape, 3) holding a 3D unit vector field.
    """
    metadata = {
        "region": {
            "name": "Extended Isocortex",
            "query": "@^\\bIsocortex|lfbs\\b$",
            "attribute": "acronym",
            "with_descendants": True,
        },
        "layers": {
            "names": [
                "layer_1",
                "layer_23",
                "layer_4",
                "layer_5",
                "layer_6",
                "lateral forebrain bundle system",
            ],
            "queries": [
                "@.*1[ab]?$",
                "@.*[2-3][ab]?$",
                "@.*4[ab]?$",
                "@.*5[ab]?$",
                "@.*6[ab]?$",
                "lfbs",
            ],
            "attribute": "acronym",
            "with_descendants": True,
        },
    }
    region_to_weight = {
        "@.*1[ab]?$": 6,
        "@.*[2-3][ab]?$": 5,
        "@.*4[ab]?$": 3,
        "@.*5[ab]?$": 2,
        "@.*6[ab]?$": 1,
        "lfbs": -2,
        "outside_of_brain": 0,
    }
    return layer_based_direction_vectors.compute_layered_region_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        metadata=metadata,
        region_to_weight=region_to_weight,
        shading_width=4,
        expansion_width=8,
        has_hemispheres=True,
    )


ISOCORTEX_ALGORITHMS: Dict[str, Callable[[RegionMap, VoxelData], NDArray[np.float32]]] = {
    "regiodesics": partial(_direction_vectors_per_region, algorithm="regiodesics"),
    "simple-blur-gradient": partial(
        _direction_vectors_per_region, algorithm="simple-blur-gradient"
    ),
    "shading-blur-gradient": _shading_gradient,
}


def compute_direction_vectors(
    region_map: RegionMap, annotation: VoxelData, algorithm: str = "regiodesics"
) -> NDArray[np.float32]:
    """Returns the direction vectors computed with the selected `algorithm`

    Arguments:
        region_map: RegionMap object with the hierarchy data
        annotation: VoxelData object containing the isocortex or a superset.
        algorithm: The algorithm to use for computing the direction vectors. Default is
            regiodesics.

    Returns:
        numpy.ndarray of shape (annotation.shape, 3) holding a 3D unit vector field.
    """
    if algorithm not in ISOCORTEX_ALGORITHMS:
        raise AtlasDirectionVectorsError(
            f"{algorithm} is not available. Choose from {list(ISOCORTEX_ALGORITHMS.keys())}"
        )

    direction_vectors = ISOCORTEX_ALGORITHMS[algorithm](region_map, annotation)

    warn_on_nan_vectors(
        direction_vectors, get_region_mask("Isocortex", annotation.raw, region_map), "Isocortex"
    )

    return direction_vectors

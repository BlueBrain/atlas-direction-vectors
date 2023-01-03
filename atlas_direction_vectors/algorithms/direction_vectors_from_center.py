"""Generic algorithm for the computation of direction vectors.

This algorithm applies to any brain region. A point is selected
(by default the center of the brain) and direction vectors are
establieshed as pointing away from said point.

This algorithm is intended to generate placeholder directions.
"""
from typing import Optional

import numpy as np
import numpy.typing
import voxcell
from atlas_commons.typing import AnnotationT, BoolArray
from atlas_commons.utils import get_region_mask
from scipy import ndimage


def command(
    region_map: voxcell.RegionMap,
    annotation: AnnotationT,
    region_accronym: str,
    center: Optional[numpy.typing.ArrayLike],
) -> np.ndarray:
    """
    Wrapper for `compute_direction_vectors` to be used in CLI.
    """
    region_mask = get_region_mask(region_accronym, annotation, region_map)

    if center is None:
        root_mask = get_region_mask("root", annotation, region_map)
        center = compute_center_of_region(root_mask)

    return compute_direction_vectors(region_mask, center)


def compute_direction_vectors(
    region: BoolArray,
    center: numpy.typing.ArrayLike,
) -> np.ndarray:
    """
    Computes within `region` direction vectors that aim away from the `center`.

    Args:
        region(numpy.ndarray): boolean 3D mask of the region.
        center(numpy.array): point in 3D marking the center.

    Returns:
        A vector field of float32 3D unit vectors over the input 3D volume.

    """
    direction_vectors = np.zeros((*region.shape, 3))
    for coordinates in np.ndindex(region.shape):
        if region[coordinates]:
            direction_vectors[coordinates] = _normalize(np.array(coordinates) - np.array(center))
    return direction_vectors


def compute_center_of_region(region: BoolArray) -> numpy.typing.ArrayLike:
    """
    Computes the center of mass for the given region.
    """
    return ndimage.center_of_mass(region)


def _normalize(item: np.ndarray) -> np.ndarray:
    """
    Normalizes array with 2-norm.
    """
    return item / np.linalg.norm(item)

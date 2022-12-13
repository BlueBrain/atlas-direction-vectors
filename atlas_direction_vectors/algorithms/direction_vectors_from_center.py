"""Generic algorithm for the computation of direction vectors.

This algorithm applies to any brain region. A point is selected
(by default the center of the brain) and direction vectors are
establieshed as pointing away from said point.

This algorithm is intended to generate placeholder directions.
"""
import numpy as np
import numpy.typing as npt
from atlas_commons.typing import BoolArray
from scipy import ndimage


def compute_direction_vectors(
    region: BoolArray,
    center: npt.ArrayLike,
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


def compute_center_of_region(region: BoolArray) -> npt.ArrayLike:
    """
    Computes the center of mass for the given region.
    """
    return ndimage.center_of_mass(region)


def _normalize(item: np.ndarray) -> np.ndarray:
    """
    Normalizes array with 2-norm.
    """
    return item / np.linalg.norm(item)

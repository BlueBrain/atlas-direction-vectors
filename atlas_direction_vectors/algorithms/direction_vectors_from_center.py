"""Generic algorithm for the computation of direction vectors.

This algorithm applies to any brain region. A point is selected
(by default the center of the brain) and direction vectors are
establieshed as pointing away from said point.

This algorithm is intended to generate placeholder directions.
"""
from typing import Optional

import numpy as np
import numpy.typing as npt
from atlas_commons.typing import BoolArray
from scipy import ndimage

CENTER_OF_MOUSE_BRAIN = (148.6458536939575, 79.81845843864052, 113.88175413924235)


def compute_direction_vectors(
    region: BoolArray,
    center: Optional[npt.ArrayLike],
) -> np.ndarray:
    if center is None:
        center = CENTER_OF_MOUSE_BRAIN

    direction_vectors = np.zeros((*region.shape, 3))
    for coordinates in np.ndindex(region.shape):
        if region[coordinates]:
            direction_vectors[coordinates] = _normalize(
                np.array(coordinates) - np.array(center)
            )
    return direction_vectors


def compute_center_of_region(region: BoolArray) -> npt.ArrayLike:
    return ndimage.center_of_mass(region)


def _normalize(item: np.ndarray) -> np.ndarray:
    return item / np.linalg.norm(item)

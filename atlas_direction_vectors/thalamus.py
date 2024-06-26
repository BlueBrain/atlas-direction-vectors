"""
Function computing the direction vectors of the mouse thalamus
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np
from atlas_commons.typing import BoolArray, NDArray
from atlas_commons.utils import get_region_mask, query_region_mask
from scipy.ndimage import correlate  # type: ignore
from scipy.ndimage import generate_binary_structure  # type: ignore
from voxcell import RegionMap, VoxelData

from atlas_direction_vectors.algorithms.layer_based_direction_vectors import (
    HemisphereOppositeOption,
    direction_vectors_for_hemispheres,
)
from atlas_direction_vectors.utils import warn_on_nan_vectors

L = logging.getLogger(__name__)
logging.captureWarnings(True)


def _get_common_outer_boundary(mask: BoolArray, sub_mask: BoolArray) -> BoolArray:
    """
    Get the mask of the voxels outside `mask` which are both
    in the outer boundary of `mask` and of `sub_mask`.

    The mask `submask`is assumed to represent a voxel subset of
    `mask`.

    Args:
        mask: boolean array of shape (W, H, D) where W, H and D are integer dimensions.
            This array holds the mask of a 3D region.
        sub_mask: boolean array of the same shape (W, H, D) as `mask`.
            mask of the voxels of a 3D sub region of the region defined by `mask`.

    Returns:
        boolean array of shape (W, H, D) representing the common outer boundary of
        `mask` and `sub_mask`.
    """
    filter_ = generate_binary_structure(3, 1).astype(int)

    return np.logical_and(
        np.logical_and(correlate(sub_mask, filter_), correlate(mask, filter_)), ~mask
    )


def compute_direction_vectors(
    region_map: Union[str, dict, RegionMap], brain_regions: VoxelData
) -> NDArray[np.float32]:
    """
    Compute the mouse thalamus direction vectors.

    Arguments:
        region_map: a RegionMap object.
        brain_regions: VoxelData object containing the mouse thalamus or a superset.

    Returns:
        Vector field of 3D unit vectors over the thalamus volume with the same shape
        as the input one. Voxels outside the thalamus have np.nan coordinates.
    """

    # For information on which regions were chosen and how this list was created, see the internal
    # BBP Confluence page located at "Circuits > Mouse Thalamus > Atlas-based Whole-thalamus
    # subregion selection".
    thalamus_query = {
        "query": "@^(?:AD|AMd|AMv|AV|CL|CM|Eth|IAD|IAM|IGL|IMD|IntG|LD|LGd-co|LGd-ip|LGd-sh|LGv_O|"
        + "LP|MD_O|MGd|MGm|MGv|PCN|PF|PIL|PO|POL|PR|PT|PVT|PoT|RE|RH|SGN|SMT|SPA|SPFm|SPFp|SubG|"
        + "TH_O|VAL|VM|VPL|VPLpc|VPM|VPMpc|Xi)$",
        "attribute": "acronym",
        "with_descendants": False,
    }

    thalamus_mask = query_region_mask(thalamus_query, brain_regions.raw, region_map)

    reticular_nucleus_mask = get_region_mask("RT", brain_regions.raw, region_map)
    reticular_nucleus_complement_mask = np.logical_and(thalamus_mask, ~reticular_nucleus_mask)
    common_outer_boundary_mask = _get_common_outer_boundary(thalamus_mask, reticular_nucleus_mask)
    landscape = {
        "source": np.zeros_like(thalamus_mask),
        "inside": reticular_nucleus_complement_mask,
        "target": common_outer_boundary_mask,
    }
    ratio = (
        brain_regions.voxel_dimensions[0] / 25
    )  # tuning based on tests with the 25 um resolution
    rt_complement_direction_vectors = direction_vectors_for_hemispheres(
        landscape,
        algorithm="simple-blur-gradient",
        hemisphere_opposite_option=(HemisphereOppositeOption.IGNORE_OPPOSITE_HEMISPHERE),
        # The use of this algorithm was initially done by Hugo Dictus
        # The constants below have been changed using trial-and-error by Austin Soplata
        sigma=ratio * 9.0,
        source_weight=0,
        target_weight=1,
        radius=100,
    )

    landscape = {
        "source": reticular_nucleus_complement_mask,
        "inside": reticular_nucleus_mask,
        "target": common_outer_boundary_mask,
    }

    rt_direction_vectors = direction_vectors_for_hemispheres(
        landscape,
        algorithm="simple-blur-gradient",
        hemisphere_opposite_option=(HemisphereOppositeOption.IGNORE_OPPOSITE_HEMISPHERE),
        # The use of this algorithm was initially done by Hugo Dictus
        # The constants below have been changed using trial-and-error by Austin Soplata
        sigma=ratio * 8.0,
        source_weight=-0.2,
        target_weight=0,
    )

    direction_vectors = np.full(rt_direction_vectors.shape, np.nan, dtype=np.float32)
    direction_vectors[reticular_nucleus_complement_mask] = rt_complement_direction_vectors[
        reticular_nucleus_complement_mask
    ]
    direction_vectors[reticular_nucleus_mask] = rt_direction_vectors[reticular_nucleus_mask]
    warn_on_nan_vectors(direction_vectors, thalamus_mask, "Thalamus")

    return direction_vectors

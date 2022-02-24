"""
Functions for interpolating invalid direction vectors by valid ones
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from atlas_commons.typing import AnnotationT, BoolArray, FloatArray, NDArray
from atlas_commons.utils import create_layered_volume, query_region_mask, split_into_halves
from voxcell import RegionMap

from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError
from atlas_direction_vectors.vector_field import interpolate

L = logging.getLogger(__name__)
logging.captureWarnings(True)


def interpolate_vectors(  # pylint: disable=too-many-locals, too-many-arguments
    annotation: AnnotationT,
    region_map: RegionMap,
    metadata: dict,
    direction_vectors: FloatArray,
    nans: bool = False,
    mask: Optional[BoolArray] = None,
    restrict_to_hemisphere: bool = False,
    restrict_to_layer: bool = False,
) -> None:
    """
    Interpolate the direction vectors of the voxels in `mask` by those of voxels out of `mask`.

    If `mask` is None, the voxel under consideration are those with a [NaN, NaN, NaN] direction
    vector.

    The interpolation is made separately on each hemisphere and on each layer if the flags
    `restrict_to_hemisphere` and `restrict_to_layer` are set to True.

    Mutate `direction_vectors` in place.

    Args:
        annotation: integer array of shape (W, H, D) enclosing the AIBS annotation of
            of a superset of the region described by `metadata`, e.g., the whole mouse brain.
            The integers W, H and D denotes the dimensions of the array.
        region_map: RegionMap object to navigate the brain regions hierarchy.
        metadata: dict, see :fun:`atlas_direction_vectors.utils.assert_metadata`.
            This dict contains the definition of the region of interest. The definition of the
            layers to be used to restrict interpolation is also required if `restrict_to_layer` is
            True.
        direction_vectors: float array of shape (W, H, D, 3) holding the direction vectors of the
            region of interest.
        nans: (Optional) a flag indicating that [NaN, NaN, NaN] direction vectors inside the region
            of interest should be interpolated.
        mask: (Optional) a boolean mask of the voxels whose direction vectors will be interpolated.
            Defaults to None, in which case the voxels with [NaN, NaN, NaN] direction vectors are
            considered for interpolation.
        restrict_to_hemisphere: If True, interpolation is performed on each hemisphere separately.
            Defaults to False.
        restrict_to_layer: If True, interpolation is performed on each layer separately.
            Defaults to False.
    """
    if mask is None and not nans:
        raise AtlasDirectionVectorsError(
            "The 'mask' argument is None and the 'nans' argument is False. At "
            "least one of them must be set."
        )

    nan_mask = np.isnan(np.linalg.norm(direction_vectors, axis=-1))
    if mask is None:
        mask = nan_mask
    if nans:
        valid_mask = np.invert(np.logical_or(mask, nan_mask))
        invalid_mask = np.invert(valid_mask)
    else:
        valid_mask = np.logical_and(np.invert(mask), np.invert(nan_mask))
        invalid_mask = mask

    if restrict_to_layer:
        layered_volume = create_layered_volume(annotation, region_map, metadata)
    else:
        layered_volume = query_region_mask(metadata["region"], annotation, region_map)

    layered_volumes: Tuple[NDArray[np.integer], ...] = (layered_volume,)
    if restrict_to_hemisphere:
        layered_volumes = split_into_halves(layered_volume)

    for layered_hemisphere in layered_volumes:
        for label in np.unique(layered_hemisphere[layered_hemisphere != 0]):
            layer_mask = layered_hemisphere == label
            valid = np.logical_and(layer_mask, valid_mask)
            invalid = np.logical_and(layer_mask, invalid_mask)
            interpolate(direction_vectors, invalid, valid, interpolator="nearest-neighbour")

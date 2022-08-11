"""Generic algorithm for the computation of direction vectors.

This algorithm applies to every brain region for which the fiber directions follow
streamlines which are orthogonal to a source and a target boundary surface.
The brain region under scrutinity must have a well-delineated laminar structure
(e.g., a series of layers).

This algorithm is used in the case of the mouse cerebellum.

The computation relies on the creation of an ad doc scalar field serving as waypoints
for the direction vectors. The user can specify scalar values for intermediate subregions
between the source (lowest values) and the target surfaces (highest values) so as
to constrain the fiber directions orthogonally to specific surface boundaries.

Constant weights are applied on the region of interest and its surroundings.
For some regions, it is also necessary to overlay the constant weight fields
with thin shadings close to specific boundaries.

A Gaussian blur is then applied to the initial scalar field and
the normalized gradient is returned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from atlas_commons.typing import AnnotationT, NDArray
from scipy.ndimage import binary_dilation, generate_binary_structure

from atlas_direction_vectors.algorithms.utils import compute_blur_gradient
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError


@dataclass
class RegionShading:
    """
    Data class holding the parameters used to define a thin scalar shading on a region boundary.

    By shading, we mean a series of values which increases with the distance to the
    boundary of interest and then vanishes beyond a specified limit.

    A region to shade is defined by means of a list of region identifiers.
    If the `invert` flag is set to True, the region to shade is the
    complement of the listed regions.

    Attrs:
        ids(list): list of region identifiers.
        boundary_region(int): identifier of a region, the boundary of which
            will be used to compute the shading.
            The `boundary_region` is the identifier of a region the boundary of which
            is used to compute the shading.
            Fibers, or equivalently direction vectors, are supposed to go out of this boundary
            before entering the region to shade.
        boundary_offset(int): smallest integer value of the shading assigned to the voxels
            which are the closest from `boundary_region`.
            It is assigned to the voxels of the region to shade
            which are the closest from the `boundary_region`.
        limit_distance(int): distance beyond which the shading stops and
            vanishes. Beyond this limit, voxels are assigned a zero value.
        invert(bool): If False, the shading's region is defined by `ids`.
            Otherwise this region is defined as the set of identifiers which don't belong to
            `ids`.
    """

    ids: list
    boundary_region: int
    boundary_offset: int
    limit_distance: int
    invert: bool = False


def shading_from_boundary(
    annotation_raw: AnnotationT, region_shading: RegionShading
) -> NDArray[np.integer]:
    """
    Computes a scalar field which increases with the distance to a region.

    This function computes an integer scalar field, similar to a distance field,
    whose value increases with the distance to a specified region boundary and becomes zero
    beyond a specified limit. The non-zero values of the field can be shifted by an
    offset parameter, namely the `boundary_offset` of the `region_shading` argument.

    The region to shade, the boundary region and the limit distance are also specified
    by `region_shading`, see RegionShading documentation for details.

    Voxels outside the region to shade are assigned a zero value.
    Voxels inside the region to shade are assigned a distance-like value, that is,
     `boundary_offset` + i if the voxel lies in the i-thick boundary
     of the region of interest but not in its (i - 1)-thick boundary.
    In particular, `boundary_offset` is the smallest positive value of the resulting shading
     and corresponds to the voxels which are the closest to the boundary region.

    Arguments:
        annotation_raw: 3D integer array holding regions identifiers.
        region_shading: RegionShading object holding the shading parameters.
            See RegionShading documentation.

    Returns:
        3D numpy.ndarray of integers, i.e., integer field over the input 3D volume.
    """
    if region_shading.limit_distance <= 0:
        raise AtlasDirectionVectorsError(
            f"Limit distance should be greater than zero : {region_shading.limit_distance}"
        )

    boundary_region = 1
    region_to_shade = 2

    region_mask = np.zeros(annotation_raw.shape, dtype=int)
    region_mask[annotation_raw == region_shading.boundary_region] = boundary_region

    region_to_shade_mask = np.isin(annotation_raw, region_shading.ids, invert=region_shading.invert)
    region_mask[region_to_shade_mask] = region_to_shade

    shades = (
        np.arange(1, region_shading.limit_distance + 1, dtype=int) + region_shading.boundary_offset
    )

    shading_mask = _sequential_region_shading(
        annotation_raw=region_mask,
        region_label=boundary_region,
        shading_target_label=region_to_shade,
        shades=shades,
    )

    return shading_mask


def _sequential_region_shading(
    annotation_raw: AnnotationT,
    region_label: int,
    shading_target_label: int,
    shades: NDArray[np.integer],
) -> NDArray[np.integer]:
    """Grows a region outwards using morphological binary dilation.

    The region that will be expanded corresponds to the voxels in `annotation_raw` that have a
    value equal to `region_label`. The process is iterative and the region grows after each
    dilation (1 voxel radius). After each dilation i, the `shades[i]` value is assigned to the
    newly explored voxels.

    Notes:
        The dilation is only allowed to grow into voxels the values of which are equal to
        `shading_target_label`.

    Arguments:
        annotation_raw: 3D integer array holding regions identifiers.
        region_label: The region of interest id.
        shading_target_label: The id of the allowed region to grow into.
        shades: The values to asign in each iteration to the dilated region.

    Returns:
        3D numpy.ndarray of integers, i.e., integer field over the input 3D volume.
    """
    region_mask = annotation_raw.copy()
    shading_mask = np.zeros_like(annotation_raw)

    for shading_value in shades:

        boundary_mask = region_dilation(
            annotation_raw=region_mask,
            region_label=region_label,
            shading_target_label=shading_target_label,
        )

        shading_mask[boundary_mask] = shading_value
        region_mask[boundary_mask] = region_label

    return shading_mask


def region_dilation(
    annotation_raw: AnnotationT, region_label: int, shading_target_label: int = 0
) -> NDArray[np.integer]:
    """Dilates selectively a region using a box of shape (3, 3, 3).

    The dilation morphological operation is applied exclusively on `annotation_raw` region, the
    voxel values of which are equal to `region_label`. The dilation is restricted in updating only
    the voxels the values of which are equal to `shading_target_label`.

    Arguments:
        annotation_raw: 3D integer array holding regions identifiers.
        region_label: The region identifier that is going to be dilated.
        shading_target_label: This label determines the allowed space to grow into.

    Returns:
        Binary mask of the same shape as `annotation_raw` the entries of which are True for the
        dilated region constrained by the allowed region.

    Notes:
        The returned mask does not include the initial region mask, only the voxels around it that
        correspond to the dilation process.
    """
    # region that we are allowed to grow into
    allowed_region = annotation_raw == shading_target_label

    # The binary dilation will be applied on the isolated layer mask
    isolated_mask = annotation_raw == region_label

    # a 3x3x3 everywhere True kernel
    struct = generate_binary_structure(3, 3)

    # the dilation we assign values only to the allowed region
    # i.e. we allow the current pass to grow as far as it does not overwrite existing voxel ids
    return binary_dilation(isolated_mask, struct) & allowed_region


def compute_initial_field(annotation_raw: AnnotationT, region_weights, shadings=()):
    """
    Initialize a scalar field based on local region fields.

    Build a scalar field whose building blocks are constant weights overlayed
    by shadings increasing from specified region boundaries.

    Constant weights are assigned to regions where the direction vectors will be
    computed but also to their surrounding regions.

    Shadings are computed on specified regions. They overlay the corresponding
    constant weight fields.

    By overlay we mean that non-zero values of shadings override
    constant weights and constant weights replace shadings zero values
    where they overlap.

    Args:
        annotation_raw (numpy.ndarray): 3D integer array holding the whole brain
            annotation.
        region_weights(dict): Dictionary with {key: value} pairs of the form
            {id: weight} where `id` is a region identifer and `weight` an integer.
        shadings(list): Optional list of RegionShading instances, see RegionShading documentation.
        Defaults to () (no shading).

    Returns:
        initial_scalar_field is an integer numpy.ndarray whose shape is
        annotation_raw.shape
    """
    initial_field = np.zeros(annotation_raw.shape, dtype=int)

    # Constant weights are set first.
    for id_, weight in region_weights.items():
        initial_field[annotation_raw == id_] = weight

    # Shadings are created and overlayed with the constant weight fields.
    for shading in shadings:
        shaded = shading_from_boundary(annotation_raw, shading)
        shading_support = shaded > 0
        initial_field[shading_support] = shaded[shading_support]

    return initial_field


def compute_direction_vectors(annotation_raw, initial_field, region_of_interest):
    """
    Computes the annotated volume's direction vectors as the normalized gradient
    of a custom scalar field.

    The direction vectors are only computed inside a region of interest.

    Direction vectors are obtained as the normalized gradient of a scalar field which resembles
    a signed distance field. This field increases in the vicinity of the target regions, i.e,
    where fibers will eventually exit or end. Such regions are assigned the highest values.
    It decreases in the vicinity of the source regions, i.e., regions from where fibers originate.
    Such regions are assigned the lowest values.

    This scalar field is first initialized by means of constant weights
    in the subregions of interest and in their surrounding regions where a
    single weight is assumed to be sufficient. For the remaining surrounding regions,
    an additional scalar shading is computed based on the distance to a subregion of
    interest identified as a target for fibers. Such a shading is overlays the
    constant weight assigned to its region.

    A Gaussian filter is then applied to the initialized scalar field
    and the normalized gradient of the blurred scalar field is eventually returned.

    The direction vectors assigned in regions which are not in the region of interest are
    3D vectors with numpy.nan coordinates.

    Arguments:
        annotation_raw(numpy.ndarray): 3D integer array holding
            the complete annotated volume, in particular the region of interest
            and its surroundings.
        initial_field(numpy.ndarray): 3D integer array holding a field defined on
            the complete annotated volume. This array has the same shape
            as `annotation_raw`.
        region_of_interest(list): list of region identifiers. Direction vectors
            will be computed for these identifiers only.


    Returns:
        float32 numpy.ndarray of shape (annotation.shape, 3), holding a
        3D vector fields of unit vectors. Outside the region of interest,
        the returned 3D vectors have numpy.nan coordinates.
    """

    # Get a smooth float field and return its gradient.
    direction_vectors = compute_blur_gradient(initial_field.astype(np.float32))

    # Direction vectors generated outside the region of interest are invalidated.
    direction_vectors[~np.isin(annotation_raw, region_of_interest), :] = np.nan

    return direction_vectors

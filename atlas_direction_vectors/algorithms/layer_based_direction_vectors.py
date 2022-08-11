"""
Generic script to compute direction vectors of a laminar region.

This script supports the case of regions separated in two hemispheres such as the isocortex or
the thalamus. It allows the use of two different lower-level algorithms computing directions
vectors. Both algorithms are based on the identification of a source and a target region:
a simple blur gradient approach and Regiodesics.

These two algorithms are appropriate when the fibers of the brain region
follow streamlines which start from and end to specific surfaces. The region
from where fibers originate is referred to as the source region.
The region where fibers end is referred to as the target region.
In Regiodesics terminology, these correspond respectively to the bottom and top
shells, a.k.a lower and upper shells.

This script is used to compute the mouse isocortex and the mouse thalamus directions vectors.
"""
from __future__ import annotations

import enum
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from atlas_commons.typing import BoolArray, FloatArray, NDArray
from atlas_commons.utils import create_layered_volume, split_into_halves
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_direction_vectors.algorithms import blur_gradient, regiodesics, simple_blur_gradient
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError

ALGORITHMS: Dict[str, Callable] = {
    "simple-blur-gradient": simple_blur_gradient.compute_direction_vectors,
    "regiodesics": regiodesics.compute_direction_vectors,
}


class HemisphereOppositeOption(enum.Enum):
    """Options for how hemipheres are handled"""

    NO_SPLIT = enum.auto()  # the region of interest is not split into hemispheres
    INCLUDE_AS_SOURCE = enum.auto()  # opposite hemisphere used as source
    INCLUDE_AS_TARGET = enum.auto()  # opposite hemisphere used as target
    IGNORE_OPPOSITE_HEMISPHERE = enum.auto()  # opposite hemisphere is not used


def attributes_to_ids(
    region_map: RegionMap,
    attributes: List[Union[Tuple[str, str], Tuple[str, int]]],
) -> List[int]:
    """
    Make a list of region identifiers out of hierarchy attributes including descendants.

    Args:
        region_map: the hierachy description of the region of interest (RegionMap object).
        attributes: list of pairs (attribute: str, value: Union[str, int])
            where `attribute` is either 'id', 'acronym' or 'name' and `value` is a value for
            of `attribute`.

    Return:
        duplicate-free list of region identifiers corresponding to the
        input attribute values.

    """
    ids = set()
    for (attribute, value) in attributes:
        ids |= region_map.find(value, attribute, ignore_case=False, with_descendants=True)
    return list(ids)


def direction_vectors_for_hemispheres(
    landscape: Dict[str, BoolArray],
    algorithm: str,
    hemisphere_opposite_option: HemisphereOppositeOption,
    **kwargs: Union[int, float, str],
) -> NDArray[np.float32]:
    """
    Compute direction vectors for each of the two hemispheres.

    Arguments:
        landscape: dict of the form
            {'source': NDArray[bool], 'inside': NDArray[bool], 'target': NDArray[bool]}
            where the value corresponding to
                'source' is the 3D binary mask of the source region, i.e.,
                    the region where the fibers originate from,
                'inside' is the 3D binary mask of the region where direction vectors
                    are computed,
                'target' is the 3D binary mask of the fibers target region.
        algorithm: the algorithm to use to generate direction vectors
                   (either 'simple-blur-gradient' or 'regiodesics').
        hemisphere_opposite_option: how the opposite hemisphere should be handled
        kwargs: (optional) Options specific to the underlying algorithm.
            For regiodesics.compute_direction_vectors, the option regiodesics_path=str can be used
            to indicate where the regiodesics executable is located. Otherwise this function will
            attempt to find it by means of distutils.spawn.find_executable.
            For simple-blur-gradient.direction_vectors, the option sigma=float can be used to
            specify the standard deviation of the Gaussian blur while source_weight=float,
            target_weight=float can be used to set custom weights in the source and target regions.

    Returns:
        Array holding a vector field of unit vectors defined on the `inside` 3D volume. The shape
        of this array is (W, L, D, 3) if the shape of `inside` is (W, L, D).
        Outside the `inside` volume, the returned direction vectors have np.nan coordinates.
    """
    assert isinstance(
        hemisphere_opposite_option, HemisphereOppositeOption
    ), f"Not a valid hemisphere_opposite_option: {hemisphere_opposite_option}"
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS.keys()}")

    hemisphere_masks = [
        landscape["inside"],
    ]
    if hemisphere_opposite_option != HemisphereOppositeOption.NO_SPLIT:
        # We assume that the region of interest has two hemispheres
        # which are symetric wrt the plane z = volume.shape[2] // 2.
        hemisphere_masks = split_into_halves(  # type: ignore
            np.ones(landscape["inside"].shape, dtype=bool)
        )

    direction_vectors = np.full(landscape["inside"].shape + (3,), np.nan, dtype=np.float32)
    for hemisphere in hemisphere_masks:
        source = (
            np.logical_or(landscape["source"], ~hemisphere)
            if hemisphere_opposite_option == HemisphereOppositeOption.INCLUDE_AS_SOURCE
            else np.logical_and(landscape["source"], hemisphere)
        )
        target = (
            np.logical_or(landscape["target"], ~hemisphere)
            if hemisphere_opposite_option == HemisphereOppositeOption.INCLUDE_AS_TARGET
            else np.logical_and(landscape["target"], hemisphere)
        )
        direction_vectors[hemisphere] = ALGORITHMS[algorithm](
            source, np.logical_and(landscape["inside"], hemisphere), target, **kwargs
        )[hemisphere]

    return direction_vectors


Attribute = Union[Tuple[str, str], Tuple[str, int]]
AttributeList = List[Attribute]


def compute_direction_vectors(
    region_map: Union[str, dict, RegionMap],
    annotation: Union[str, VoxelData],
    landscape: Dict[str, AttributeList],
    algorithm: str = "simple-blur-gradient",
    hemisphere_opposite_option: HemisphereOppositeOption = HemisphereOppositeOption.NO_SPLIT,
    **kwargs: Union[int, float, str],
) -> NDArray[np.float32]:
    """
    Computes within `inside` direction vectors that originate from `source` and end in `target`.

    Args:
        region_map: a path to hierarchy.json or dict made of such a file or a
            RegionMap object. Defaults to None.
        annotation: full annotation array from which the region of interest `inside` will be
            extracted.
        landscape: landscape: dict of the form
            {source': AttributeList, 'inside': AttributeList, 'target': AttributeList}
            where the value corresponding to
                'source' is a list of acronyms or of integer identifiers defining the
                    source region of fibers.
                'inside' is a list of acronyms or of integer identifiers defining the
                    the region where the direction vectors are computed.
                'target' is a list of acronyms or of integer identifiers defining the
                    the region where the fibers end.
        algorithm: name of the algorithm to be used for the computation
            of direction vectors. One of `regiodesics` or `simple-blur-gradient`.
            Defaults to `simple-blur-gradient`.
        hemisphere_opposite_option: how the opposite hemisphere should be handled
        kwargs: see direction_vectors_for_hemispheres documentation.

    Returns:
        A vector field of float32 3D unit vectors over the input 3D volume.

    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"`algorithm` must be one of {ALGORITHMS}")

    if isinstance(annotation, str):
        annotation = VoxelData.load_nrrd(annotation)
    else:
        if not isinstance(annotation, VoxelData):
            raise ValueError("`annotation` must be specified as a path or a VoxelData object.")

    new_landscape = {
        "source": np.isin(
            annotation.raw,
            attributes_to_ids(region_map, landscape["source"]),
        ),
        "inside": np.isin(
            annotation.raw,
            attributes_to_ids(region_map, landscape["inside"]),
        ),
        "target": np.isin(
            annotation.raw,
            attributes_to_ids(region_map, landscape["target"]),
        ),
    }
    direction_vectors = direction_vectors_for_hemispheres(
        new_landscape, algorithm, hemisphere_opposite_option, **kwargs
    )

    return direction_vectors


# pylint: disable=too-many-locals
def compute_layered_region_direction_vectors(
    region_map: RegionMap,
    annotation: VoxelData,
    metadata: dict,
    region_to_weight: Dict[str, int],
    shading_width: int,
    expansion_width: int,
    has_hemispheres: bool = False,
) -> FloatArray:
    """
    Calculates the direction vectors in the layered region determined by `metadata`.

    The goal is to compute a vector field pointing towards a general direction, based on a source
    region and a target region. Within the region of interest, direction vectors are obtained as
    the normalized gradient of a scalar field. This scalar field is obtained by assigning to every
    voxel of the brain a user-defined weight representing its distance from the source region.

    For each subregion of interest, a single weight is assigned to every voxel of that subregion
    and a default value is given to the rest of the brain. To avoid boundary effects from the
    outside at the borders of a region, we extend the scalar field of the region to its surrounding
    voxels. These surrounding voxels are detected by a shading algorithm which looks for voxels
    close to annotation borders (i.e. where a change of annotation is occuring). We apply this
    shading algorithm to each layer of the Isocortex, setting their surrounding voxels to the same
    weight in the scalar field.

    An additional scalar shading is computed based on the distance to a subregion of interest
    identified as a target for fibers. This shading is created to attract the gradient of the
    voxels in the target region towards the outside. For the Isocortex, voxels closed to the L1 and
    outside of the brain are assigned to 6 plus their distance to L1.

    A Gaussian filter is then used to the initialized scalar field and the gradient of the
    normalized blurred scalar field is eventually returned. The direction vectors are given by this
    gradient. This process is applied to all cortical areas by defining the white matter as the
    source region, and the outside of the brain as the target.

    Notes:
        The last region in the `metadata[layers][queries]` is assumed to correspond to an external
        group of regions that will be ommited from the final orientation field.

    Arguments:
        region_map: RegionMap instance
        annotation: Full annotation array from which the region of interest `inside` will be
        metadata: dict describing the region of interest and its layers.
        region_to_weight: dict the keys of which are acronyms and the values weight integers
        shading_width:
        expansion_width: The number of times to apply the region dilation on all layers.
        has_hemispheres: If true it splits the volume into two hemispheres, processing each one
            independently.

    Returns:
        A vector field of float32 3D unit vectors over the input 3D volume.

    Notes:
        The parameter choice for shading_width and expansion_width are 4 and 8 respectively.
        The blurring helps to spread the weights that are assigned to the fields. For the weights
        of the shading (border_region) there is a strong gradient with the ouside (0 -> >10), which
        means there is a strong gradient inwards the region despite the width of the shading. To
        fix this the shading size must be reduced but should remain greater than sigma=3 so that
        inside the region it will not affect by the outside, hence 4. Then to prevent a strong
        gradient inwards on the shading the annotation is extended by size of shading:
        expansion_width = shading_width + sigma + 1 = 8
    """
    layer_queries = metadata["layers"]["queries"]

    if not set(layer_queries).issubset(set(region_to_weight.keys())):
        raise AtlasDirectionVectorsError(
            f"Layer queries are not included in the region_to_weight dict\n"
            f"Layer queries: {layer_queries}\n"
            f"region_to_weight={region_to_weight}"
        )

    layered_region = create_layered_volume(annotation.raw, region_map, metadata)

    ids = np.unique(layered_region)
    if len(ids[ids != 0]) != len(layer_queries):
        raise AtlasDirectionVectorsError(
            f"Layer region ids were not correctly assigned from the layer_queries\n"
            f"Layered region ids: {ids}\n"
            f"layer queries: {layer_queries}"
        )

    # example: ids[ids!=0] -> [1, 2, 3], layers -> [1, 2], external_id -> 3
    *layers, external_id = ids[ids != 0]

    layer_to_weight = _build_layered_region_weights(layer_queries, region_to_weight)

    # make a mask separating the first layer from the rest
    border_region_mask = np.zeros(annotation.raw.shape, dtype=np.uint8)
    border_region_mask[layered_region > 0] = 1
    border_region_mask[layered_region == 1] = 2

    shading_complement = blur_gradient.RegionShading(
        ids=[1, 2],
        boundary_region=2,
        boundary_offset=layer_to_weight[1],
        limit_distance=shading_width,
        invert=True,
    )

    direction_vectors = np.full(annotation.raw.shape + (3,), np.nan, dtype=np.float32)

    if has_hemispheres:

        for hemisphere_mask in split_into_halves(np.full(annotation.raw.shape, True, dtype=bool)):
            layered_hemisphere = np.zeros_like(hemisphere_mask, dtype=np.uint8)
            np.copyto(layered_hemisphere, layered_region, where=hemisphere_mask)

            hemi_direction_vectors = _expanded_boundary_shading(
                layered_hemisphere,
                layers,
                layer_to_weight,
                border_region_mask,
                shading_complement,
                expansion_width,
            )

            direction_vectors[hemisphere_mask, :] = hemi_direction_vectors[hemisphere_mask, :]

    else:

        direction_vectors[:] = _expanded_boundary_shading(
            layered_region,
            layers,
            layer_to_weight,
            border_region_mask,
            shading_complement,
            expansion_width,
        )

    # remove the grown regions into the void from the dilation and the external_id values
    # (fibers) that are not considered in the final field.
    direction_vectors[(layered_region == 0) | (layered_region == external_id)] = np.nan
    return direction_vectors


def _build_layered_region_weights(
    regions: List[str], region_to_weight: Dict[str, int]
) -> Dict[int, int]:
    """Creates layer identifiers for each region in `regions`, starting at 1 and maps them to
    their respective region weights.

    Notes:
        The special "outside_of_brain" key is converted to a layer_id 0.

    Returns:
        Dictionary the keys of which are layer ids or zero and the values are int weights.
    """
    layer_to_weight = {
        layer_id: region_to_weight[region] for layer_id, region in enumerate(regions, start=1)
    }

    if "outside_of_brain" in region_to_weight:
        layer_to_weight[0] = region_to_weight["outside_of_brain"]

    return layer_to_weight


def _expanded_boundary_shading(
    layered_region: NDArray[np.integer],
    layers: List[int],
    layer_to_weight: Dict[int, int],
    border_region_mask: NDArray[np.integer],
    shading_complement: blur_gradient.RegionShading,
    expansion_width,
) -> NDArray[np.integer]:
    """Implementation of the compute_layered_region_direction_vectors algorithm.

    Arguments:
        layered_region: 3D array with integer values corresponding to the layers of the region,
            starting at 1, and 0 corresponding to the outside of the brain.
        layers: List of integer layers, starting at 1
        layer_to_weight: Dictionary the keys of which are layer ids and the values of which are
            gradient weights.
        border_region_mask: Integer mask with 2 corresponding to the boundary region, 1 to the rest
            of the regions and 0 to the void.
        shading_complement: RegionShading object for the complement of the region.
        expansion_width: The number of times to apply the region dilation on all layers.

    Returns:
        A vector field of float32 3D unit vectors over the input 3D volume.
    """

    layered_region = layered_region.copy()

    # the loop below updates the layered_region in place, therefore we need to keep track of the
    # initial region
    initial_region_mask = layered_region != 0

    # the expansion of the boundary is applied 5 times so that a big enough region is created
    # to prevent the influence of the outside in the vector field calculation.
    for _ in range(expansion_width):
        for layer in layers[::-1]:
            layered_region[
                blur_gradient.region_dilation(
                    annotation_raw=layered_region, region_label=layer, shading_target_label=0
                )
            ] = layer

    field = blur_gradient.compute_initial_field(layered_region, layer_to_weight)

    region_mask = np.zeros_like(layered_region, dtype=np.int8)
    np.copyto(region_mask, border_region_mask, where=initial_region_mask)

    shading_border = blur_gradient.shading_from_boundary(region_mask, shading_complement)

    shading_mask: BoolArray = np.logical_and(
        shading_border > 0, np.logical_or(field == layer_to_weight[1], field == layer_to_weight[0])
    )

    np.copyto(field, shading_border, where=shading_mask)

    return blur_gradient.compute_direction_vectors(layered_region > 0, field, layers)

"""Generic tools for finding direction vectors in a region"""
import numpy as np
from atlas_commons.utils import get_region_mask
from voxcell.math_utils import minimum_aabb  # pylint: disable=ungrouped-imports

from atlas_direction_vectors.algorithms.layer_based_direction_vectors import (
    compute_direction_vectors,
    compute_layered_region_direction_vectors,
)


def layered_region(annotation, region_map, outside_of_brain, layer_weights, has_hemispheres):
    """wrapper for compute_layered_region_direction_vectors so not all metadata is needed"""
    # convert to metadata
    query = "@" + "|".join(rf"\b{acronym}\b" for acronym in layer_weights)
    metadata = {
        "region": {
            "name": "NOT USED",
            "query": query,
            "attribute": "acronym",
            "with_descendants": True,
        },
        "layers": {
            "names": list(layer_weights),
            "queries": list(layer_weights),
            "attribute": "acronym",
            "with_descendants": True,
        },
    }

    region_to_weight = layer_weights.copy()
    region_to_weight["outside_of_brain"] = outside_of_brain

    return compute_layered_region_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        metadata=metadata,
        region_to_weight=region_to_weight,
        shading_width=4,
        expansion_width=8,
        has_hemispheres=has_hemispheres,
    )


def source_target_layered_region(
    annotation, region_map, algorithm, source, region, target, hemisphere_opposite_option
):
    """Computes within `inside` direction vectors from `source` to `target`."""
    direction_vectors = np.full(annotation.shape + (3,), np.nan, dtype=np.float32)

    region_mask = get_region_mask(region, annotation.raw, region_map)
    # pylint: disable=not-an-iterable
    aabb_slice = tuple(
        slice(bottom, top + 1) for (bottom, top) in np.array(minimum_aabb(region_mask)).T
    )
    voxel_data = annotation.with_data(annotation.raw[aabb_slice])
    region_direction_vectors = compute_direction_vectors(
        region_map,
        voxel_data,
        {
            "source": [("acronym", source)],
            "inside": [("acronym", region)],
            "target": [("acronym", target)],
        },
        algorithm,
        hemisphere_opposite_option,
    )

    region_mask = region_mask[aabb_slice]
    direction_vectors[aabb_slice][region_mask] = region_direction_vectors[region_mask]

    return direction_vectors

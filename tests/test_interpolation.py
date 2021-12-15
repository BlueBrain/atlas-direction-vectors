"""
Unit tests for the interplation of direction vectors
"""
import numpy as np
import numpy.testing as npt
import pytest as pt
from voxcell import RegionMap

import atlas_direction_vectors.interpolation as tested
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError


def get_hierarchy():
    return {
        "id": 0,
        "acronym": "B",
        "name": "Brain",
        "children": [
            {"id": id_, "acronym": f"B{id_}", "name": f"Region{id_}", "children": []}
            for id_ in range(1, 9)
        ],
    }


def get_region_map():
    return RegionMap.from_dict(get_hierarchy())


def get_metadata(queries=None):
    if queries is None:
        queries = ["@^Region[1|3|5|7]$", "@^Region[2|4|6|8]$"]

    return {
        "region": {
            "name": "Brain",
            "query": "Brain",
            "attribute": "name",
            "with_descendants": True,
        },
        "layers": {
            "names": ["layer_1", "layer_2"],
            "queries": queries,
            "attribute": "name",
            "with_descendants": True,
        },
    }


def get_direction_vectors():
    direction_vectors = np.ones((2, 2, 2, 3), dtype=float)
    direction_vectors[0, 0, 0] = [np.nan, np.nan, np.nan]
    direction_vectors[1, 1, 1] = [np.nan, np.nan, np.nan]

    return direction_vectors


def test_interpolate_nan_vectors():
    annotation = np.arange(1, 9).reshape((2, 2, 2))
    direction_vectors = get_direction_vectors()

    tested.interpolate_vectors(
        annotation,
        get_region_map(),
        get_metadata(),
        direction_vectors,
        nans=True,
        mask=None,
    )
    npt.assert_array_equal(direction_vectors, np.ones((2, 2, 2, 3), dtype=float))


def test_interpolate_exception():
    annotation = np.arange(1, 9).reshape((2, 2, 2))
    direction_vectors = get_direction_vectors()

    with pt.raises(AtlasDirectionVectorsError):
        tested.interpolate_vectors(
            annotation,
            get_region_map(),
            get_metadata(),
            direction_vectors,
            nans=False,
            mask=None,
        )


def test_interpolate_vectors_with_mask():
    annotation = np.arange(1, 9).reshape((2, 2, 2))
    direction_vectors = get_direction_vectors()

    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    expected = np.ones((2, 2, 2, 3), dtype=float)
    expected[1, 1, 1] = [np.nan] * 3

    tested.interpolate_vectors(
        annotation,
        get_region_map(),
        get_metadata(),
        direction_vectors,
        mask=mask,
    )
    npt.assert_array_equal(direction_vectors, expected)


def test_interpolate_nan_vectors_restrict_to_hemisphere():
    annotation = np.arange(1, 9).reshape((2, 2, 2))
    direction_vectors = np.ones((2, 2, 2, 3), dtype=float)
    direction_vectors[:, :, 1] = 2
    direction_vectors[0, 0, 0] = [np.nan, np.nan, np.nan]
    direction_vectors[1, 1, 1] = [np.nan, np.nan, np.nan]

    expected = direction_vectors.copy()
    expected[0, 0, 0] = [1, 1, 1]
    expected[1, 1, 1] = [2, 2, 2]

    tested.interpolate_vectors(
        annotation,
        get_region_map(),
        get_metadata(),
        direction_vectors,
        nans=True,
        mask=None,
        restrict_to_hemisphere=True,
    )
    npt.assert_array_equal(direction_vectors, expected)


def test_interpolate_nan_vectors_restrict_to_layer():
    annotation = np.arange(1, 9).reshape((2, 2, 2))
    direction_vectors = np.ones((2, 2, 2, 3), dtype=float)
    direction_vectors[:, :, 1] = 2
    direction_vectors[0, 0, 0] = [np.nan, np.nan, np.nan]
    direction_vectors[1, 1, 1] = [np.nan, np.nan, np.nan]

    expected = direction_vectors.copy()
    expected[0, 0, 0] = [1, 1, 1]
    expected[1, 1, 1] = [2, 2, 2]

    tested.interpolate_vectors(
        annotation,
        get_region_map(),
        get_metadata(),
        direction_vectors,
        nans=True,
        mask=None,
        restrict_to_layer=True,
    )
    npt.assert_array_equal(direction_vectors, expected)


def get_input_data():
    direction_vectors = np.ones((2, 2, 2, 3), dtype=float)
    direction_vectors[:, :, 1] = 2
    direction_vectors[0, 1, 0] = [3, 3, 3]
    direction_vectors[1, 0, 1] = [3, 3, 3]
    direction_vectors[0, 0, 0] = [np.nan, np.nan, np.nan]
    direction_vectors[1, 1, 1] = [np.nan, np.nan, np.nan]
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True

    return {
        "annotation": np.arange(1, 9).reshape((2, 2, 2)),
        "direction_vectors": direction_vectors,
        "hierarchy": get_hierarchy(),
        "metadata": get_metadata(["@^Region[1|2|3|4]$", "@^Region[5|6|7|8]$"]),
        "mask": mask,
    }


def test_interpolate_nan_vectors_restrict_to_hemisphere_and_layer():
    data = get_input_data()
    expected = data["direction_vectors"].copy()
    expected[0, 0, 0] = [3, 3, 3]
    expected[1, 1, 1] = [3, 3, 3]

    tested.interpolate_vectors(
        data["annotation"],
        get_region_map(),
        data["metadata"],
        data["direction_vectors"],
        nans=True,
        mask=None,
        restrict_to_hemisphere=True,
        restrict_to_layer=True,
    )
    npt.assert_array_equal(data["direction_vectors"], expected)


def test_interpolate_with_mask_restrict_to_hemisphere_and_layer():
    data = get_input_data()
    expected = data["direction_vectors"].copy()
    expected[0, 0, 0] = [3, 3, 3]
    expected[1, 1, 1] = [np.nan] * 3

    tested.interpolate_vectors(
        data["annotation"],
        get_region_map(),
        data["metadata"],
        data["direction_vectors"],
        mask=data["mask"],
        restrict_to_hemisphere=True,
        restrict_to_layer=True,
    )
    npt.assert_array_equal(data["direction_vectors"], expected)


def test_interpolate_with_mask_and_nans_options():
    data = get_input_data()
    expected = data["direction_vectors"].copy()
    expected[0, 0, 0] = [3, 3, 3]
    expected[1, 1, 1] = [3, 3, 3]

    tested.interpolate_vectors(
        data["annotation"],
        get_region_map(),
        data["metadata"],
        data["direction_vectors"],
        nans=True,
        mask=data["mask"],
        restrict_to_hemisphere=True,
        restrict_to_layer=True,
    )
    npt.assert_array_equal(data["direction_vectors"], expected)

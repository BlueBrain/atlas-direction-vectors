import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from voxcell import RegionMap, VoxelData

import atlas_direction_vectors.isocortex as tested
from tests.algorithms.test_layer_based_direction_vectors import check_direction_vectors
from tests.mark import skip_if_no_regiodesics

TEST_PATH = Path(Path(__file__).parent)
HIERARCHY_PATH = str(Path(TEST_PATH, "1.json"))
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def region_map():
    return RegionMap.load_json(HIERARCHY_PATH)


def _check_vectors_direction_dominance(direction_vectors, annotation, direction):

    region_mask = annotation.raw > 983

    region_vectors = direction_vectors[region_mask, :]

    # First check that all vectors are in the same halfspace as the direction
    cosines = region_vectors.dot(direction)
    assert np.all(cosines > 0.0), f"Angles to direction:\n{np.rad2deg(np.arccos(cosines))}"

    # Then check that they form an angle less that 45 degrees with the direction
    mask = np.arccos(cosines) < 0.25 * np.pi
    assert mask.all(), (
        f"Less than 45: {np.sum(mask)}, More than 45: {np.sum(~mask)}\n"
        f"Angles to direction:\n{np.rad2deg(np.arccos(cosines))}"
    )


def test_get_isocortical_regions(region_map):
    raw = np.arange(1, 35).reshape((1, 2, 17))
    expected = ["SSp-m", "SSp-tr", "VISp"]
    # Path to hierarchy.json
    regions = tested.get_isocortical_regions(raw, region_map)
    npt.assert_array_equal(regions, expected)

    # True RegionMap object
    regions = tested.get_isocortical_regions(raw, region_map)
    npt.assert_array_equal(regions, expected)


@skip_if_no_regiodesics
def test_compute_direction_vectors(region_map):
    # Two high-level regions, namely ACAd and ACAv
    # with layers 1, 2/3, 5 and 6
    raw = np.zeros((16, 16, 16), dtype=int)
    # ACAd6
    raw[3:8, 3:12, 3] = 927  # ACAd6b, since ACAd6a is ignored
    raw[3:8, 3:12, 12] = 927
    # ACAd5
    raw[3:8, 3:12, 4] = 1015
    raw[3:8, 3:12, 11] = 1015
    # ACAd2/3
    raw[3:8, 3:12, 5:7] = 211
    raw[3:8, 3:12, 9:11] = 211
    # ACAd1
    raw[3:8, 3:12, 7:9] = 935

    # ACAv6
    raw[8:12, 3:12, 3] = 819  # ACAv6b since ACAv6a is ignored
    raw[8:12, 3:12, 12] = 819
    # ACAv5
    raw[8:12, 3:12, 4] = 772
    raw[8:12, 3:12, 11] = 772
    # ACAv2/3
    raw[8:12, 3:12, 5:7] = 296
    raw[8:12, 3:12, 9:11] = 296
    # ACAv1
    raw[8:12, 3:12, 7:9] = 588

    voxel_data = VoxelData(raw, (1.0, 1.0, 1.0))
    direction_vectors = tested.compute_direction_vectors(
        region_map, voxel_data, algorithm="regiodesics"
    )
    check_direction_vectors(direction_vectors, raw > 0, {"opposite": "target", "strict": False})

    direction_vectors = tested.compute_direction_vectors(
        region_map, voxel_data, algorithm="simple-blur-gradient"
    )
    check_direction_vectors(direction_vectors, raw > 0, {"opposite": "target", "strict": False})


def test_compute_direction_vectors_with_missing_bottom(region_map):
    # Two high-level regions, namely ACAd and ACAv
    # with layers 1, 2/3, 5
    # Layer 6 is missing and troubles are expected!
    raw = np.zeros((16, 16, 16), dtype=int)

    # ACAd5
    raw[3:8, 3:12, 4] = 1015
    raw[3:8, 3:12, 11] = 1015
    # ACAd2/3
    raw[3:8, 3:12, 5:7] = 211
    raw[3:8, 3:12, 9:11] = 211
    # ACAd1
    raw[3:8, 3:12, 7:9] = 935

    # ACAv5
    raw[8:12, 3:12, 4] = 772
    raw[8:12, 3:12, 11] = 772
    # ACAv2/3
    raw[8:12, 3:12, 5:7] = 296
    raw[8:12, 3:12, 9:11] = 296
    # ACAv1
    raw[8:12, 3:12, 7:9] = 588

    voxel_data = VoxelData(raw, (1.0, 1.0, 1.0))
    with warnings.catch_warnings(record=True) as w:
        tested.compute_direction_vectors(region_map, voxel_data)
        assert "NaN" in str(w[-1].message)


def test_compute_directions__shading_gradient(region_map):

    raw = np.zeros((20, 30, 20), dtype=np.int32)

    raw[6:14, (6, 7), 6:14] = 983
    raw[6:14, (8, 9), 6:14] = 12998
    raw[6:14, (10, 11), 6:14] = 12997
    raw[6:14, (12, 13), 6:14] = 12996
    raw[6:10, (14, 15), 6:14] = 12995
    raw[10:14, (14, 15), 6:14] = 12994
    raw[6:14, (16, 17), 6:14] = 12994
    raw[6:14, (16, 17), 6:14] = 12994
    raw[6:14, (18, 19), 6:14] = 12993

    annotation = VoxelData(raw, [25.0, 25.0, 25.0])

    direction_vectors = tested.compute_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        algorithm="shading-blur-gradient",
    )

    _check_vectors_direction_dominance(direction_vectors, annotation, np.array([0.0, 1.0, 0.0]))

    # above and below void -> nan
    assert np.all(np.isnan(direction_vectors[:, :6, :]))
    assert np.all(np.isnan(direction_vectors[:, 20:, :]))

    # outside roi void -> nan
    assert np.all(np.isnan(direction_vectors[:14, :, 14:]))

    # fibers 983 should not be present in the final directions -> nan
    assert np.all(np.isnan(direction_vectors[:, 0:8, :]))

    # the rest should not be nan
    assert not np.any(np.isnan(direction_vectors[6:14, 8:20, 6:14]))

    expected_direction_vectors = VoxelData.load_nrrd(
        DATA_DIR / "isocortex_shading_gradient_orientations.nrrd"
    )

    npt.assert_allclose(direction_vectors, expected_direction_vectors.raw)

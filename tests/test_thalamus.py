"""
Unit tests for the creation of the thalamus direction vectors
"""
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from voxcell import RegionMap, VoxelData

import atlas_direction_vectors.thalamus as tested

TEST_PATH = Path(Path(__file__).parent)
HIERARCHY_PATH = str(Path(TEST_PATH, "1.json"))


@pytest.fixture
def region_map():
    return RegionMap.load_json(HIERARCHY_PATH)


def create_voxeldata(th_id_1: int, th_id_2: int):
    """
    Create cube missing the slice x == 1.

    RT is represented by the lower part.

    Args:
        th_id_1: a thalamic region id distinct from 262 (RT)
        th_id_2: a thalamic region id distinct from 262 (RT)

    Returns:
        VoxelData object capturing some features of the actual
        AIBS CCFv3 thalamus volume.
    """
    raw = np.zeros((10, 10, 10), dtype=int)
    raw[2:, :5, :] = th_id_1
    raw[2:, 4:, :] = th_id_2
    raw[1, :, :] = 0
    raw[0, :, :] = 262  # RT, Reticular nucleus of the thalamus
    # The final array shape is (12, 12, 12)
    raw = np.pad(raw, 1, "constant", constant_values=0)  #  1-voxel of padding

    return VoxelData(raw, (2, 2, 2))


def test_get_common_outer_boundary():
    mask = np.ones((5, 5, 5), dtype=bool)
    mask[2, ...] = False  # Creating
    sub_mask = np.zeros_like(mask)
    sub_mask[1, ...] = True

    # The final array shape is (7, 7, 7)
    mask = np.pad(mask, 1, "constant", constant_values=False)
    sub_mask = np.pad(sub_mask, 1, "constant", constant_values=False)

    expected = np.zeros_like(mask)
    expected[3, 1:6, 1:6] = True
    expected[2, [0, 6], 1:6] = True
    expected[2, 1:6, [0, 6]] = True

    actual = tested._get_common_outer_boundary(mask, sub_mask)
    npt.assert_array_equal(actual, expected)


def test_compute_direction_vectors(region_map):
    voxel_data = create_voxeldata(
        685,  # VM, Ventral medial nucleus of the thalamus
        709,  # VP, Ventral posterior complex of the thalamus
    )
    with warnings.catch_warnings(record=True) as warnings_:
        direction_vectors = tested.compute_direction_vectors(region_map, voxel_data)
        assert not warnings_
        assert np.all(
            np.isnan(direction_vectors[2, 1:11, 1:11])
        )  # out-of-thalamus vectors should be (nan, nan, nan)
        assert np.all(np.isnan(direction_vectors[[0, 11], :, :]))
        assert np.all(np.isnan(direction_vectors[:, [0, 11], :]))
        assert np.all(np.isnan(direction_vectors[:, :, [0, 11]]))
        norms = np.linalg.norm(direction_vectors, axis=-1)
        npt.assert_allclose(norms[3:11, 1:11, 1:11], 1.0, rtol=1e-6)
        npt.assert_allclose(norms[1, 1:11, 1:11], 1.0, rtol=1e-6)

    # Reducing the voxel dimensions reduces the Gaussian blur standard deviation.
    # With a low standard deviation, the gradient of the blur can be zero and its normalization
    # is therefore (nan, nan, nan).
    voxel_data.voxel_dimensions = (1, 1, 1)
    with warnings.catch_warnings(record=True) as w:
        tested.compute_direction_vectors(region_map, voxel_data)
        assert "NaN" in str(w[-1].message)

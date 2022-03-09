from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

from atlas_direction_vectors.algorithms import regiodesics as tested
from atlas_direction_vectors.algorithms.regiodesics import RegiodesicsLabels
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError
from tests.mark import skip_if_no_regiodesics


@patch(
    "atlas_direction_vectors.algorithms.regiodesics.find_executable",
    return_value="/home/software/Regiodesics/layer_segmenter",
)
def test_find_regiodesics_exec_or_raise_found(_):
    assert (
        tested.find_regiodesics_exec_or_raise("layer_segmenter")
        == "/home/software/Regiodesics/layer_segmenter"
    )


@patch(
    "atlas_direction_vectors.algorithms.regiodesics.find_executable",
    return_value="",
)
def test_find_regiodesics_exec_or_raise_raises(_):
    with pytest.raises(FileExistsError):
        tested.find_regiodesics_exec_or_raise("geodesics")


def test_mark_with_regiodesics_labels():
    full_volume = np.zeros((9, 9, 9), dtype=int)
    full_volume[:, :, :3] = 1  # bottom
    full_volume[:, :, 3:6] = 2  # in between
    full_volume[:, :, 6:] = 3  # top
    marked = tested.mark_with_regiodesics_labels(
        full_volume == 1, full_volume == 2, full_volume == 3
    )
    expected = np.zeros((9, 9, 9), dtype=int)
    expected[:, :, 4] = RegiodesicsLabels.INTERIOR
    expected[:, :, 3] = RegiodesicsLabels.BOTTOM
    expected[0, :, 4] = RegiodesicsLabels.SHELL
    expected[8, :, 4] = RegiodesicsLabels.SHELL
    expected[:, 0, 4] = RegiodesicsLabels.SHELL
    expected[:, 8, 4] = RegiodesicsLabels.SHELL
    expected[:, :, 5] = RegiodesicsLabels.TOP
    npt.assert_array_equal(marked, expected)


def test_mark_with_regiodesics_labels_exception():
    full_volume = np.zeros((9, 9, 9), dtype=int)
    full_volume[:, :, 3:6] = 2  # in between
    full_volume[:, :, 6:] = 3  # top
    with pytest.raises(AtlasDirectionVectorsError) as error:
        tested.mark_with_regiodesics_labels(full_volume == 1, full_volume == 2, full_volume == 3)
    assert "Empty bottom" in str(error.value)

    full_volume[:, :, :3] = 1  # bottom
    full_volume[:, :, 6:] = 0  # empty top
    with pytest.raises(AtlasDirectionVectorsError) as error:
        tested.mark_with_regiodesics_labels(full_volume == 1, full_volume == 2, full_volume == 3)
    assert "Empty top" in str(error.value)


@skip_if_no_regiodesics
def test_compute_direction_vectors():
    raw = np.zeros((8, 8, 8), dtype=int)
    raw[:, :, :2] = 1  # bottom
    raw[:, :, 2:6] = 2  # interior
    raw[:, :, 6:8] = 3  # top
    direction_vectors = tested.compute_direction_vectors(raw == 1, raw == 2, raw == 3)
    expected = np.zeros(raw.shape + (3,))
    expected[:, :, :] = np.array([0.0, 0.0, 1.0])
    nan_mask = np.isnan(direction_vectors)
    npt.assert_array_almost_equal(direction_vectors[~nan_mask], expected[~nan_mask])


def test_compute_direction_vectors_exception():
    raw = np.zeros((8, 8, 8), dtype=int)
    raw[:, :, 2:6] = 2  # interior
    raw[:, :, 6:8] = 3  # top
    with pytest.raises(AtlasDirectionVectorsError):
        tested.compute_direction_vectors(raw == 1, raw == 2, raw == 3)

"""
Unit tests of the utility functions used for direction vectors computations.
"""
import warnings

import numpy as np

import atlas_direction_vectors.utils as tested


def test_warn_on_nan_vectors():
    mask = np.ones((1, 1, 1), dtype=bool)
    direction_vectors = np.ones((1, 1, 1, 3), dtype=float)

    with warnings.catch_warnings(record=True) as w:
        tested.warn_on_nan_vectors(direction_vectors, mask, "Thalamus")
        assert not w

    direction_vectors[0, 0, 0] = [np.nan] * 3
    with warnings.catch_warnings(record=True) as w:
        tested.warn_on_nan_vectors(direction_vectors, mask, "Thalamus")
        assert len(w) == 2
        assert isinstance(w[0].message, UserWarning)
        assert "(NaN, NaN, NaN) direction vectors in 100.000% of the Thalamus voxels" in str(
            w[0].message
        )
        assert isinstance(w[1].message, UserWarning)
        assert "Consider interpolating (NaN, NaN, NaN) vectors by valid ones." in str(w[1].message)

    with warnings.catch_warnings(record=True) as w:
        tested.warn_on_nan_vectors(direction_vectors, mask, "Isocortex")
        assert len(w) == 3
        assert "(NaN, NaN, NaN) direction vectors in 100.000% of the Isocortex voxels" in str(
            w[0].message
        )
        assert (
            "(NaN, NaN, NaN) direction vectors are likely to prevent you from splitting layer 2/3."
            in str(w[1].message)
        )
        assert isinstance(w[2].message, UserWarning)
        assert "Consider interpolating (NaN, NaN, NaN) vectors by valid ones." in str(w[2].message)

    mask = np.zeros((1, 1, 1), dtype=bool)
    with warnings.catch_warnings(record=True) as w:
        tested.warn_on_nan_vectors(direction_vectors, mask, "Isocortex")
        assert len(w) == 0

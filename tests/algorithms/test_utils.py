import numpy as np
import numpy.testing as npt
import pytest as pyt
from atlas_commons.utils import normalized

import atlas_direction_vectors.algorithms.utils as tested


def test_compute_blur_gradient():
    scalar_field = np.zeros((5, 5, 5))
    scalar_field[0, :, :] = -1
    scalar_field[1, :, :] = 1

    # The standard deviation  of the Gaussian blur is large enough, so that
    # the gradient is non-zero everywhere.
    gradient = tested.compute_blur_gradient(scalar_field)  # the default stddev is 3.0
    assert np.all(~np.isnan(gradient))
    assert np.all(gradient[..., 0] > 0.0)  # vectors flow along the positive x-axis
    npt.assert_array_almost_equal(np.linalg.norm(gradient, axis=3), np.full((5, 5, 5), 1.0))

    # The standard deviation of the Gaussian blur is too small:
    # some gradient vectors are zero, but the non-zero ones
    # are normalized as expected.
    gradient = tested.compute_blur_gradient(scalar_field, 0.1)
    nan_mask = np.isnan(gradient)
    assert np.any(nan_mask)
    norm = np.linalg.norm(gradient, axis=3)
    npt.assert_array_almost_equal(norm[~np.isnan(norm)], np.full((5, 5, 5), 1.0)[~np.isnan(norm)])

    # Wrong input type
    scalar_field = np.ones((2, 2, 2), dtype=int)
    with pyt.raises(ValueError):
        tested.compute_blur_gradient(scalar_field, 0.1)


class Test_vector_to_quaternion:
    @pyt.mark.xfail
    def test_long_vector_gives_same_quat(self):
        npt.assert_almost_equal(
            tested.vector_to_quaternion(np.array([[[1.0, 0.0, 0.0]]])),
            tested.vector_to_quaternion(np.array([[[5, 0, 0]]])),
        )


def test_compute_boundary():
    v_1 = np.zeros((5, 5, 5), dtype=bool)
    v_1[1:4, 1:4, 1:4] = True
    v_2 = ~v_1
    boundary = tested.compute_boundary(v_1, v_2)
    expected = np.copy(v_1)
    expected[2, 2, 2] = False
    npt.assert_array_equal(boundary, expected)

    v_1 = np.zeros((5, 5, 5), dtype=bool)
    v_1[0:2, :, 1:4] = True
    v_2 = np.zeros_like(v_1)
    v_2[2:, :, 1:4] = True
    boundary = tested.compute_boundary(v_1, v_2)
    expected = np.zeros_like(v_1)
    expected[1, :, 1:4] = True
    npt.assert_array_equal(boundary, expected)

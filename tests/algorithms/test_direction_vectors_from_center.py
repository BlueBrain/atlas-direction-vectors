import numpy as np
import pytest

from atlas_direction_vectors.algorithms.direction_vectors_from_center import (
    compute_center_of_region,
    compute_direction_vectors,
)


def test_compute_direction_vectors():
    region = np.array(
        (
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
        )
    )
    center = (0.5, 1, 1)
    assert np.allclose(
        compute_direction_vectors(region, center),
        np.array(
            [
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [-0.4472136, 0.0, 0.89442719],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [-0.4472136, 0.89442719, 0.0],
                        [-0.33333333, 0.66666667, 0.66666667],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.4472136, 0.0, 0.89442719],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.4472136, 0.89442719, 0.0],
                        [0.33333333, 0.66666667, 0.66666667],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ],
            ],
        ),
    )


@pytest.mark.parametrize(
    "region, expected_result",
    [
        (
            np.array(
                (
                    [
                        [1, 0],
                        [0, 0],
                    ],
                    [
                        [0, 0],
                        [0, 0],
                    ],
                )
            ),
            (0.0, 0.0, 0.0),
        ),
        (
            np.array(
                (
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                    ],
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                    ],
                )
            ),
            (0.5, 2.5, 1.5),
        ),
    ],
)
def test_compute_center_of_region(region, expected_result):
    assert compute_center_of_region(region) == expected_result

import numpy as np
import numpy.testing as npt
import pytest

from atlas_direction_vectors.algorithms import blur_gradient as tested
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError


def _print_arrays(*arrays):
    class Colors:

        RED = "\u001b[31m"
        CYAN = "\u001b[36m"
        GREEN = "\u001b[32m"
        WHITE = "\u001B[37m"
        YELLOW = "\u001b[33m"
        PURPLE = "\u001B[35m"
        BLACK = "\u001B[30m"
        BLUE = "\u001B[33m"
        END = "\033[0m"

    def row2str(row):
        return " ".join([f"{Colors.END}{colors[i]}{i:>2}{Colors.END}" for i in row])

    colors = {
        0: Colors.WHITE,
        1: Colors.CYAN,
        2: Colors.GREEN,
        3: Colors.BLUE,
        4: Colors.YELLOW,
        5: Colors.PURPLE,
    }
    for i in range(6, 10000):
        colors[i] = Colors.WHITE

    string = ""

    for planes in zip(*arrays):
        for rows in zip(*planes):
            string += "\t\t".join(map(row2str, rows)) + "\n"
        string += "\n"

    return string


@pytest.fixture
def annotation():

    raw = np.zeros((6, 8, 8), dtype=np.int32)

    raw[(1, 2), 2:6, 2:6] = 1
    raw[(3, 4), 2:6, 2:6] = 2

    return raw


def test_region_dilation(annotation):

    actual_mask = tested.region_dilation(annotation, region_label=1, shading_target_label=0)

    expected_mask = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=np.int32,
    )

    npt.assert_array_equal(
        actual_mask.astype(int),
        expected_mask.astype(int),
        err_msg=_print_arrays(actual_mask.astype(int), expected_mask.astype(int)),
    )

    actual_mask = tested.region_dilation(annotation, region_label=2, shading_target_label=0)

    expected_mask = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=np.int32,
    )

    npt.assert_array_equal(
        actual_mask.astype(int),
        expected_mask.astype(int),
        err_msg=_print_arrays(actual_mask.astype(int), expected_mask.astype(int)),
    )


def test_sequential_region_shading(annotation):

    res = tested._sequential_region_shading(
        annotation, region_label=2, shading_target_label=0, shades=[5, 4]
    )

    expected_shading = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 0, 0, 0, 0, 4, 4],
                [4, 4, 0, 0, 0, 0, 4, 4],
                [4, 4, 0, 0, 0, 0, 4, 4],
                [4, 4, 0, 0, 0, 0, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ],
            [
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ],
            [
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],  # 2 was here
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ],
            [
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],  # 2 was here
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 0, 0, 0, 0, 5, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ],
            [
                [4, 4, 4, 4, 4, 4, 4, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 5, 5, 5, 5, 5, 5, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ],
        ],
        dtype=np.int32,
    )

    npt.assert_array_equal(res, expected_shading, err_msg=_print_arrays(res, expected_shading))


def test_shading_from_boundary():

    input_mask = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=np.int32,
    )

    # negative limit distance is not valid
    shading = tested.RegionShading(
        ids=[0, 1], boundary_region=1, boundary_offset=0, limit_distance=-1, invert=True
    )
    with pytest.raises(AtlasDirectionVectorsError):
        tested.shading_from_boundary(input_mask, shading)

    # zero limit distance is not valid
    shading = tested.RegionShading(
        ids=[0, 1], boundary_region=1, boundary_offset=0, limit_distance=0, invert=True
    )
    with pytest.raises(AtlasDirectionVectorsError):
        tested.shading_from_boundary(input_mask, shading)

    # roi not present in the input_mask
    shading = tested.RegionShading(
        ids=[0, 3], boundary_region=3, boundary_offset=0, limit_distance=1, invert=True
    )
    assert np.all(tested.shading_from_boundary(input_mask, shading) == 0)

    # all regions are ignored
    shading = tested.RegionShading(
        ids=[0, 1, 2], boundary_region=1, boundary_offset=0, limit_distance=1, invert=True
    )
    assert np.all(tested.shading_from_boundary(input_mask, shading) == 0)

    expected = np.zeros(input_mask.shape, dtype=int)
    expected[3, 1:5, 1:5] = 1
    shading = tested.RegionShading(
        ids=[0, 1], boundary_region=1, boundary_offset=0, limit_distance=1, invert=True
    )
    actual = tested.shading_from_boundary(input_mask, shading)
    npt.assert_array_equal(actual, expected, err_msg=_print_arrays(input_mask, actual, expected))

    expected[4, 1:5, 1:5] = 2
    shading = tested.RegionShading(
        ids=[0, 1], boundary_region=1, boundary_offset=0, limit_distance=2, invert=True
    )
    actual = tested.shading_from_boundary(input_mask, shading)
    npt.assert_array_equal(actual, expected, err_msg=_print_arrays(input_mask, actual, expected))

    shading = tested.RegionShading(
        ids=[0, 1], boundary_region=1, boundary_offset=0, limit_distance=10, invert=True
    )
    actual = tested.shading_from_boundary(input_mask, shading)
    npt.assert_array_equal(actual, expected, err_msg=_print_arrays(input_mask, actual, expected))

    shading = tested.RegionShading(
        ids=[1], boundary_region=1, boundary_offset=0, limit_distance=10, invert=True
    )
    actual = tested.shading_from_boundary(input_mask, shading)

    expected = np.array(
        [
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
            ],
            [
                [3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3],
            ],
        ]
    )

    npt.assert_array_equal(actual, expected, err_msg=_print_arrays(input_mask, actual, expected))


def test_compute_initial_field_single_weights():
    # Two regions
    raw = np.zeros((2, 2, 2))
    raw[0, :, :] = 111
    raw[1, :, :] = 222
    region_weights = {111: 1, 222: 1}
    initial_field = tested.compute_initial_field(raw, region_weights)
    npt.assert_array_equal(initial_field, np.ones((2, 2, 2)))

    # Four regions
    raw = np.zeros((4, 2, 2))
    raw[0, :, :] = 111
    raw[1, :, :] = 222
    raw[2, :, :] = 333
    raw[3, :, :] = 444
    region_weights = {111: 1, 222: -1, 333: 2, 444: 2}
    initial_field = tested.compute_initial_field(raw, region_weights)
    expected_field = np.zeros_like(raw)
    expected_field[0, :, :] = 1
    expected_field[1, :, :] = -1
    expected_field[2, :, :] = 2
    expected_field[3, :, :] = 2
    npt.assert_array_equal(initial_field, expected_field)


def test_compute_initial_field_with_shadings():
    raw = np.zeros((6, 2, 2))
    raw[0, :, :] = 111
    raw[1, :, :] = 222  # to shade
    raw[2, :, :] = 333  # single weight only
    raw[3, :, :] = 444  # single weight only
    raw[4, :, :] = 555  # to shade
    raw[5, :, :] = 666  # to shade
    region_weights = {333: -1, 444: 1}
    shadings = [
        # offset = 1, limit_distance = 3, which is greater than the actual space on the left
        tested.RegionShading([111, 222], 333, 1, 3),
        # offset = 1, limit_distance = 1, which is less than the actual space on the right
        tested.RegionShading([555, 666], 444, 2, 1),
    ]
    expected_field = np.zeros_like(raw)
    expected_field[0, :, :] = 3
    expected_field[1, :, :] = 2
    expected_field[2, :, :] = -1
    expected_field[3, :, :] = 1
    expected_field[4, :, :] = 3
    expected_field[5, :, :] = 0
    initial_field = tested.compute_initial_field(raw, region_weights, shadings)
    npt.assert_array_equal(initial_field, expected_field)
    # Same field as before, but the regions to shade
    # are specified by means of their complement
    shadings = [
        # Inverted ids selection
        tested.RegionShading([333, 444, 555, 666], 333, 1, 3, invert=True),
        # Inverted ids selection
        tested.RegionShading([111, 222, 333, 444], 444, 2, 1, invert=True),
    ]
    initial_field = tested.compute_initial_field(raw, region_weights, shadings)
    npt.assert_array_equal(initial_field, expected_field)
    # Same annotation as before, but we specify additional single weights
    region_weights = {
        # The following two weights won't have any effect, since the shadings
        # are positive in these regions
        111: 6,
        222: 6,
        333: -1,
        444: 1,
        555: 6,
        666: 6,
    }
    shadings = [
        # Inverted ids selection
        tested.RegionShading([333, 444, 555, 666], 333, 1, 3, invert=True),
        # Inverted ids selection
        tested.RegionShading([111, 222, 333, 444], 444, 2, 1, invert=True),
    ]
    expected_field = np.zeros_like(raw)
    expected_field[0, :, :] = 3
    expected_field[1, :, :] = 2
    expected_field[2, :, :] = -1
    expected_field[3, :, :] = 1
    expected_field[4, :, :] = 3
    expected_field[5, :, :] = 6
    initial_field = tested.compute_initial_field(raw, region_weights, shadings)
    npt.assert_array_equal(initial_field, expected_field)


def test_compute_direction_vectors():
    raw = np.zeros((7 * 3, 2, 2))
    raw[0:3, :, :] = 111  # non-roi, fixed weight
    raw[3:6, :, :] = 222  # non-roi, fixed weight
    raw[6:9, :, :] = 333  # roi, single weight only
    raw[9:12, :, :] = 444  # roi, single weight only
    raw[12:15, :, :] = 555  # roi, single weight only
    raw[15:18, :, :] = 666  # non-roi, to shade
    raw[18:21, :, :] = 777  # non-roi, to shade, plus overlayed fixed weight
    region_weights = {111: -5, 222: -5, 333: -1, 444: 0, 555: 1, 777: 5}
    shadings = [
        tested.RegionShading([666, 777], 555, 1, 3),
    ]
    initial_field = tested.compute_initial_field(raw, region_weights, shadings)
    direction_vectors = tested.compute_direction_vectors(raw, initial_field, [333, 444, 555])
    # Direction vectors are [np.nan] * 3 for every voxel outside the
    # the regions of interest.
    region_of_interest_mask = np.isin(raw, [333, 444, 555])
    assert np.all(np.isnan(direction_vectors[~region_of_interest_mask]))
    # Inside the regions of interest, the non-nan direction vectors
    # should be unit vectors.
    norm = np.linalg.norm(direction_vectors, axis=3)
    npt.assert_array_equal(norm[~np.isnan(norm)], np.full(raw.shape, 1.0)[~np.isnan(norm)])
    # Direction vectors should flow along the positive x-axis
    assert np.all(direction_vectors[region_of_interest_mask, 0] > 0.0)

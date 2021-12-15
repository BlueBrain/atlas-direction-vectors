from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from voxcell import RegionMap, VoxelData

from atlas_direction_vectors import cerebellum as tested
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError

TEST_PATH = Path(Path(__file__).parent)
HIERARCHY_PATH = str(Path(TEST_PATH, "1.json"))
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def region_map():
    return RegionMap.load_json(HIERARCHY_PATH)


@pytest.fixture
def annotation():
    """Cerebellum toy annotation.

    The lingula (left) is side to side with the flocculus (right).

    Above both there is void and bellow them fiber tracts. On
    the sides other regions have been places so that void does
    not affect the gradient.
    """
    raw = np.array(
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
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 10707, 10707, 10707, 10692, 10692, 10692, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 10706, 10706, 10706, 10691, 10691, 10691, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 10705, 10705, 10705, 10690, 10690, 10690, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ],
            [
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
            ],
            [
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
                [728, 744, 728, 744, 728, 744, 728, 744],
            ],
        ],
        dtype=np.int32,
    )

    return VoxelData(raw, (25.0, 25.0, 25.0), offset=(1.0, 2.0, 3.0))


def _acronyms_to_flattened_identifiers(region_map, acronyms):
    ids = set()
    for acronym in acronyms:
        ids |= region_map.find(acronym, attr="acronym", with_descendants=True)
    return list(ids)


def _check_vectors_defined_in_regions(direction_vectors, region_map, annotation, acronyms):

    assert direction_vectors.shape == annotation.raw.shape + (3,)

    # The region of interest should not have nan value
    region_mask = np.isin(annotation.raw, _acronyms_to_flattened_identifiers(region_map, acronyms))
    assert not np.isnan(direction_vectors[region_mask]).any()

    # Output direction vectors should by unit vectors
    npt.assert_allclose(np.linalg.norm(direction_vectors, axis=3)[region_mask], 1.0, atol=1e-6)

    # Outside the region of interest everything should be nan
    region_mask = np.isin(
        annotation.raw, _acronyms_to_flattened_identifiers(region_map, acronyms), invert=True
    )
    assert np.isnan(direction_vectors[region_mask]).all()


def _check_vectors_direction_dominance(
    direction_vectors, region_map, annotation, acronyms, direction
):

    region_mask = np.isin(annotation.raw, _acronyms_to_flattened_identifiers(region_map, acronyms))

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


def test_compute_cerebellum_direction_vectors(region_map, annotation):

    res = tested.compute_direction_vectors(region_map, annotation)
    _check_vectors_defined_in_regions(
        res, region_map, annotation, ["FLgr", "FLpu", "FLmo"] + ["LINGgr", "LINGpu", "LINGmo"]
    )

    _check_vectors_direction_dominance(
        res, region_map, annotation, ["FLgr", "FLpu", "FLmo"], [-1.0, 0.0, 0.0]
    )

    _check_vectors_direction_dominance(
        res, region_map, annotation, ["LINGgr", "LINGpu", "LINGmo"], [-1.0, 0.0, 0.0]
    )

    expected_direction_vectors = VoxelData.load_nrrd(
        DATA_DIR / "cerebellum_shading_gradient_orientations.nrrd"
    )

    npt.assert_allclose(res, expected_direction_vectors.raw)


def test_flocculus_direction_vectors(region_map, annotation):

    res = tested._flocculus_direction_vectors(region_map, annotation)
    _check_vectors_defined_in_regions(res, region_map, annotation, ["FLgr", "FLpu", "FLmo"])


def test_lingula_direction_vectors(region_map, annotation):

    res = tested._lingula_direction_vectors(region_map, annotation)
    _check_vectors_defined_in_regions(res, region_map, annotation, ["LINGgr", "LINGpu", "LINGmo"])

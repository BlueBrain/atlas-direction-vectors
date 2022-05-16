import json
from tempfile import NamedTemporaryFile

import numpy as np
import numpy.testing as npt
import pytest
from voxcell import RegionMap, VoxelData

from atlas_direction_vectors.algorithms import layer_based_direction_vectors as tested
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError
from tests.mark import skip_if_no_regiodesics


class Test_attributes_to_ids:
    def setup_method(self):
        self.region_map = RegionMap.from_dict(
            {
                "id": 0,
                "acronym": "root",
                "name": "root",
                "children": [
                    {"id": 16, "acronym": "Isocortex", "name": "Isocortex"},
                    {"id": 22, "acronym": "CB", "name": "Cerebellum"},
                    {
                        "id": 1,
                        "acronym": "TMv",
                        "name": "Tuberomammillary nucleus, ventral part",
                    },
                    {
                        "id": 23,
                        "acronym": "TH",
                        "name": "Thalamus",
                        "children": [
                            {
                                "id": 13,
                                "acronym": "VAL",
                                "name": "Ventral anterior-lateral complex of the thalamus",
                            }
                        ],
                    },
                ],
            }
        )

    def test_mixed_attributes(self):
        attributes = [
            ("id", 1),
            ("name", "Isocortex"),
            ("id", 2),
            ("id", 2),
            ("id", 22),
            ("id", 3),
            ("id", 3),
            ("acronym", "TH"),
            ("acronym", "CB"),
        ]
        ids = tested.attributes_to_ids(self.region_map, attributes)
        npt.assert_array_equal(sorted(ids), [1, 13, 16, 22, 23])
        attributes = [
            ("id", 1),
            ("name", "Isocortex"),
            ("id", 2),
            ("id", 2),
            ("id", 22),
            ("id", 3),
            ("id", 3),
            ("name", "Cerebellum"),
        ]
        ids = tested.attributes_to_ids(self.region_map, attributes)
        npt.assert_array_equal(sorted(ids), [1, 16, 22])


def check_direction_vectors(direction_vectors, inside, options=None):
    norm = np.linalg.norm(direction_vectors, axis=3)
    # Regiodesics can produce NaN vectors in the region of interest
    # (in its boundary).
    # We take this into account by specifying a non-strict NaN policy.
    if options is not None and not options.get("strict", True):
        inside = np.logical_and(~np.isnan(norm), inside)
    assert np.all(~np.isnan(direction_vectors[inside, :]))
    half = inside.shape[2] // 2
    bottom_hemisphere = np.copy(inside)
    bottom_hemisphere[:, :, half:] = False

    if options is None or options["opposite"] == "target":
        # Vectors in the bottom hemisphere flow along the positive z-axis
        assert np.all(direction_vectors[bottom_hemisphere, 2] > 0.0)
    else:
        # Vectors in the top hemisphere flow along the negative z-axis
        assert np.all(direction_vectors[bottom_hemisphere, 2] < 0.0)
    top_hemisphere = np.copy(inside)
    top_hemisphere[:, :, :half] = False

    if options is None or options["opposite"] == "target":
        # Vectors in the top hemisphere flow along the negative z-axis
        assert np.all(direction_vectors[top_hemisphere, 2] < 0.0)
    else:
        # Vectors in the bottom hemisphere flow along the positive z-axis
        assert np.all(direction_vectors[top_hemisphere, 2] > 0.0)

    # NaNs are expected outside `inside`
    assert np.all(np.isnan(norm[~inside]))
    # Non-NaN direction vectors have unit norm
    npt.assert_array_almost_equal(norm[inside], np.full(inside.shape, 1.0)[inside])


class Test_direction_vectors_for_hemispheres:
    @staticmethod
    def landscape_1():
        source = np.zeros((16, 16, 16), dtype=bool)
        source[3:15, 3:15, 3:7] = True
        inside = np.zeros_like(source)
        inside[3:15, 3:15, 6:10] = True
        target = np.zeros_like(source)
        target[3:15, 3:15, 9:13] = True
        return {"source": source, "inside": inside, "target": target}

    @staticmethod
    def landscape_2():
        source = np.zeros((16, 16, 16), dtype=bool)
        source[3:13, 3:13, 3] = True
        source[3:13, 3:13, 12] = True
        inside = np.zeros_like(source)
        inside[3:13, 3:13, 3:13] = True
        target = np.zeros_like(source)
        target[3:13, 3:13, 7:9] = True
        return {"source": source, "inside": inside, "target": target}

    @staticmethod
    def landscape_3():
        target = np.zeros((16, 16, 16), dtype=bool)
        target[3:13, 3:13, 3] = True
        target[3:13, 3:13, 12] = True
        inside = np.zeros_like(target)
        inside[3:13, 3:13, 3:13] = True
        source = np.zeros_like(target)
        source[3:13, 3:13, 7:9] = True
        return {"source": source, "inside": inside, "target": target}

    def test_invalid_option(self):
        with pytest.raises(AssertionError):
            tested.direction_vectors_for_hemispheres(
                self.landscape_1(), "simple-blur-gradient", hemisphere_opposite_option="asdf"
            )

    def test_simple_blur_without_hemispheres(self):
        l1 = self.landscape_1()
        direction_vectors = tested.direction_vectors_for_hemispheres(
            l1,
            "simple-blur-gradient",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.NO_SPLIT,
        )
        inside = l1["inside"]
        assert np.all(~np.isnan(direction_vectors[inside, :]))
        assert np.all(direction_vectors[inside, 2] > 0.0)  # vectors flow along the positive z-axis
        norm = np.linalg.norm(direction_vectors, axis=3)
        # NaN are expected outside `inside`
        assert np.all(np.isnan(norm[~inside]))
        # Non-NaN direction vectors have unit norm
        npt.assert_array_almost_equal(norm[inside], np.full((16, 16, 16), 1.0)[inside])

    @skip_if_no_regiodesics
    def test_regiodesics_without_hemispheres(self):
        l1 = self.landscape_1()
        direction_vectors = tested.direction_vectors_for_hemispheres(
            l1, "regiodesics", hemisphere_opposite_option=tested.HemisphereOppositeOption.NO_SPLIT
        )
        inside = l1["inside"]
        assert np.all(~np.isnan(direction_vectors[inside, :]))
        assert np.all(direction_vectors[inside, 2] > 0.0)  # vectors flow along the positive z-axis
        norm = np.linalg.norm(direction_vectors, axis=3)
        # NaN are expected outside `inside`
        assert np.all(np.isnan(norm[~inside]))
        # Non-NaN direction vectors have unit norm
        npt.assert_array_almost_equal(norm[inside], np.full((16, 16, 16), 1.0)[inside])

    def test_simple_blur_with_hemispheres_no_opposite(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_2(),
            "simple-blur-gradient",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.IGNORE_OPPOSITE_HEMISPHERE,
        )
        check_direction_vectors(direction_vectors, self.landscape_2()["inside"])

    @skip_if_no_regiodesics
    def test_regiodesics_with_hemispheres_no_opposite(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_2(),
            "regiodesics",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.IGNORE_OPPOSITE_HEMISPHERE,
        )
        check_direction_vectors(direction_vectors, self.landscape_2()["inside"])

    def test_simple_blur_with_opposite_hemisphere_as_target(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_2(),
            "simple-blur-gradient",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.INCLUDE_AS_TARGET,
        )
        check_direction_vectors(
            direction_vectors, self.landscape_2()["inside"], {"opposite": "target"}
        )

    @skip_if_no_regiodesics
    def test_regiodesics_with_opposite_hemisphere_as_target(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_2(),
            "regiodesics",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.INCLUDE_AS_TARGET,
        )
        check_direction_vectors(
            direction_vectors, self.landscape_2()["inside"], {"opposite": "target"}
        )

    def test_simple_blur_with_opposite_hemisphere_as_source(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_3(),
            "simple-blur-gradient",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.INCLUDE_AS_SOURCE,
        )
        check_direction_vectors(
            direction_vectors, self.landscape_3()["inside"], {"opposite": "source"}
        )

    @skip_if_no_regiodesics
    def test_regiodesics_with_opposite_hemisphere_as_source(self):
        direction_vectors = tested.direction_vectors_for_hemispheres(
            self.landscape_3(),
            "regiodesics",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.INCLUDE_AS_SOURCE,
        )
        check_direction_vectors(
            direction_vectors, self.landscape_3()["inside"], {"opposite": "source"}
        )


class Test_compute_direction_vectors:
    @staticmethod
    def fake_hierarchy_json():
        return RegionMap.from_dict(
            {
                "id": 0,
                "children": [
                    {"id": 1},
                    {"id": 2},
                    {"id": 3},
                    {"id": 4},
                    {"id": 5},
                    {"id": 6},
                ],
            }
        )

    @staticmethod
    def voxel_data_1():
        raw = np.zeros((16, 16, 16), dtype=int)
        raw[3:15, 3:15, 3:6] = 1
        raw[3:15, 3:15, 6] = 2
        raw[3:15, 3:15, 7] = 3
        raw[3:15, 3:15, 8] = 4
        raw[3:15, 3:15, 9:12] = 5
        return VoxelData(raw, (1.0, 1.0, 1.0))

    @staticmethod
    def landscape_1():
        return {
            "source": [("id", 1), ("id", 2)],
            "inside": [("id", 1), ("id", 2), ("id", 3), ("id", 4), ("id", 5)],
            "target": [("id", 4), ("id", 5)],
        }

    @staticmethod
    def voxel_data_2():
        raw = np.zeros((16, 16, 16), dtype=int)
        raw[3:13, 3:13, 3] = 1
        raw[3:13, 3:13, 4:7] = 2
        raw[3:13, 3:13, 7] = 3
        raw[3:13, 3:13, 8] = 4
        raw[3:13, 3:13, 9:12] = 5
        raw[3:13, 3:13, 12] = 6
        return VoxelData(raw, (1.0, 1.0, 1.0))

    @staticmethod
    def landscape_2():
        return {
            "source": [("id", 1), ("id", 6)],
            "inside": [
                ("id", 1),
                ("id", 2),
                ("id", 3),
                ("id", 4),
                ("id", 5),
                ("id", 6),
            ],
            "target": [("id", 3), ("id", 4)],
        }

    @staticmethod
    def landscape_3():
        return {
            "source": [("id", 3), ("id", 4)],
            "inside": [
                ("id", 1),
                ("id", 2),
                ("id", 3),
                ("id", 4),
                ("id", 5),
                ("id", 6),
            ],
            "target": [("id", 1), ("id", 6)],
        }

    def test_raises_on_wrong_input(self):
        with pytest.raises(ValueError):
            tested.compute_direction_vectors(self.fake_hierarchy_json(), [], self.landscape_1())

        with pytest.raises(ValueError):
            tested.compute_direction_vectors(
                self.fake_hierarchy_json(),
                self.voxel_data_1(),
                self.landscape_1(),
                algorithm="unknown_algorithm",
            )

    def test_simple_blur_without_hemispheres(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(), self.voxel_data_1(), self.landscape_1()
        )
        ids = tested.attributes_to_ids(self.fake_hierarchy_json(), self.landscape_1()["inside"])
        inside = np.isin(self.voxel_data_1().raw, ids)
        assert np.all(~np.isnan(direction_vectors[inside, :]))
        assert np.all(direction_vectors[inside, 2] > 0.0)  # vectors flow along the positive z-axis
        norm = np.linalg.norm(direction_vectors, axis=3)
        # NaNs are expected outside `inside`
        assert np.all(np.isnan(norm[~inside]))
        # Non-NaN direction vectors have unit norm
        npt.assert_array_almost_equal(norm[inside], np.full((16, 16, 16), 1.0)[inside])

    def test_simple_blur_without_hemispheres_from_file(self):
        with NamedTemporaryFile(mode="w") as nrrd_file:
            self.voxel_data_1().save_nrrd(nrrd_file.name)
            direction_vectors = tested.compute_direction_vectors(
                self.fake_hierarchy_json(), nrrd_file.name, self.landscape_1()
            )
            ids = tested.attributes_to_ids(
                self.fake_hierarchy_json(),
                self.landscape_1()["inside"],
            )
            inside = np.isin(self.voxel_data_1().raw, ids)
            assert np.all(~np.isnan(direction_vectors[inside, :]))
            assert np.all(
                direction_vectors[inside, 2] > 0.0
            )  # vectors flow along the positive z-axis
            norm = np.linalg.norm(direction_vectors, axis=3)
            # NaN are expected outside `inside`
            assert np.all(np.isnan(norm[~inside]))
            # Non-NaN Direction vectors have unit norm
            npt.assert_array_almost_equal(norm[inside], np.full((16, 16, 16), 1.0)[inside])

    def test_simple_blur_with_hemispheres_no_opposite(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(),
            self.voxel_data_2(),
            self.landscape_2(),
            "simple-blur-gradient",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.IGNORE_OPPOSITE_HEMISPHERE,
        )
        ids = tested.attributes_to_ids(
            self.fake_hierarchy_json(),
            self.landscape_2()["inside"],
        )
        inside = np.isin(self.voxel_data_2().raw, ids)
        check_direction_vectors(direction_vectors, inside)

    def test_simple_blur_with_opposite_hemisphere_as_target(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(),
            self.voxel_data_2(),
            self.landscape_2(),
            "simple-blur-gradient",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.INCLUDE_AS_TARGET,
        )
        ids = tested.attributes_to_ids(
            self.fake_hierarchy_json(),
            self.landscape_2()["inside"],
        )
        inside = np.isin(self.voxel_data_2().raw, ids)
        check_direction_vectors(direction_vectors, inside, {"opposite": "target"})

    def test_simple_blur_with_opposite_hemisphere_as_source(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(),
            self.voxel_data_2(),
            self.landscape_3(),
            "simple-blur-gradient",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.INCLUDE_AS_SOURCE,
        )
        ids = tested.attributes_to_ids(
            self.fake_hierarchy_json(),
            self.landscape_3()["inside"],
        )
        inside = np.isin(self.voxel_data_2().raw, ids)
        check_direction_vectors(direction_vectors, inside, {"opposite": "source"})

    @skip_if_no_regiodesics
    def test_regiodesics_with_opposite_hemisphere_as_source(self):
        direction_vectors = tested.compute_direction_vectors(
            self.fake_hierarchy_json(),
            self.voxel_data_2(),
            self.landscape_3(),
            "regiodesics",
            hemisphere_opposite_option=tested.HemisphereOppositeOption.INCLUDE_AS_SOURCE,
        )
        ids = tested.attributes_to_ids(
            self.fake_hierarchy_json(),
            self.landscape_2()["inside"],
        )
        inside = np.isin(self.voxel_data_2().raw, ids)
        check_direction_vectors(direction_vectors, inside, {"opposite": "source"})


def test_build_layered_region_weights():

    regions = ["r1", "r2", "r3"]
    region_to_weight = {"r1": -1, "r2": 0, "r3": 1, "outside_of_brain": -2}

    res = tested._build_layered_region_weights(regions, region_to_weight)

    assert res == {0: -2, 1: -1, 2: 0, 3: 1}


def test_compute_layered_direction_vectors():

    metadata = {"layers": {"queries": ["q1", "q2"]}}
    region_to_weight = {"q1": 1, "q3": 2}

    shading_width = 6
    expansion_width = 5

    # Layer queries are not included in the weight dict
    with pytest.raises(AtlasDirectionVectorsError):
        tested.compute_layered_region_direction_vectors(
            None, None, metadata, region_to_weight, shading_width, expansion_width
        )

    region_map = RegionMap.from_dict(
        {
            "id": 0,
            "acronym": "root",
            "name": "root",
            "children": [
                {"id": 16, "acronym": "Isocortex", "name": "Isocortex"},
                {"id": 22, "acronym": "CB", "name": "Cerebellum"},
                {
                    "id": 1,
                    "acronym": "TMv",
                    "name": "Tuberomammillary nucleus, ventral part",
                },
                {
                    "id": 23,
                    "acronym": "TH",
                    "name": "Thalamus",
                    "children": [
                        {
                            "id": 13,
                            "acronym": "VAL",
                            "name": "Ventral anterior-lateral complex of the thalamus",
                        }
                    ],
                },
            ],
        }
    )

    metadata = {
        "region": {
            "name": "root",
            "query": "root",
            "attribute": "acronym",
            "with_descendants": True,
        },
        "layers": {
            "names": ["1", "2"],
            "queries": ["TMv", "Car"],
            "attribute": "acronym",
            "with_descendants": True,
        },
    }

    region_to_weight = {"TMv": -10, "TH": 10, "Isocortex": 20, "outside_of_brain": 3}

    raw = np.zeros((10, 20, 10), dtype=np.int32)

    raw[(3, 4), ...] = 1
    raw[(5, 6), ...] = 23
    raw[(7, 8), ...] = 16

    brain_regions = VoxelData(raw, (25.0, 25.0, 25.0))

    # Layer region is not converted to id
    with pytest.raises(AtlasDirectionVectorsError):
        tested.compute_layered_region_direction_vectors(
            region_map, brain_regions, metadata, region_to_weight, shading_width, expansion_width
        )

    metadata["layers"]["names"] = ["1", "2", "3"]
    metadata["layers"]["queries"] = ["TMv", "TH", "Isocortex"]

    res = tested.compute_layered_region_direction_vectors(
        region_map,
        brain_regions,
        metadata,
        region_to_weight,
        shading_width,
        expansion_width,
        has_hemispheres=False,
    )

    np.allclose(res[(3, 4, 5, 6, 7, 8), ...], [1.0, 0.0, 0.0])

    res = tested.compute_layered_region_direction_vectors(
        region_map,
        brain_regions,
        metadata,
        region_to_weight,
        shading_width,
        expansion_width,
        has_hemispheres=True,
    )

    np.allclose(res[(3, 4, 5, 6, 7, 8), ...], [1.0, 0.0, 0.0])

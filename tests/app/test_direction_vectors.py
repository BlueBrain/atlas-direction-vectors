"""test app/direction_vectors"""
import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_direction_vectors.app.direction_vectors as tested
from tests.test_interpolation import get_input_data
from tests.test_thalamus import create_voxeldata

TEST_PATH = Path(Path(__file__).parent.parent)
HIERARCHY_PATH = str(Path(TEST_PATH, "1.json"))


def test_thalamus():
    runner = CliRunner()
    with runner.isolated_filesystem():
        voxel_data = create_voxeldata(
            718,  # VPL, Ventral posterolateral nucleus of the thalamus
            709,  # VPM, Ventral posteromedial nucleus of the thalamus
        )
        voxel_data.save_nrrd("annotation.nrrd")
        result = runner.invoke(
            tested.thalamus,
            [
                "--annotation-path",
                "annotation.nrrd",
                "--hierarchy-path",
                HIERARCHY_PATH,
                "--output-path",
                "direction_vectors.nrrd",
            ],
        )
        assert result.exit_code == 0
        direction_vectors = VoxelData.load_nrrd("direction_vectors.nrrd")
        npt.assert_array_equal(direction_vectors.raw.shape, (12, 12, 12, 3))
        assert direction_vectors.raw.dtype == np.float32


def test_cerebellum():

    runner = CliRunner()
    with runner.isolated_filesystem():

        cerebellum_raw = np.zeros((8, 8, 8))
        for x_index, region_id in enumerate([10707, 10692, 10706, 10691, 10705, 10690, 744, 728]):
            cerebellum_raw[x_index, :, :] = region_id
        cerebellum_raw = np.pad(
            cerebellum_raw, 2, "constant", constant_values=0
        )  # Add 2-voxel void margin around the positive annotations

        annotation = VoxelData(cerebellum_raw, (25.0, 25.0, 25.0), offset=(1.0, 2.0, 3.0))
        annotation.save_nrrd("cerebellum_annotation.nrrd")

        result = runner.invoke(
            tested.cerebellum,
            [
                "--annotation-path",
                "cerebellum_annotation.nrrd",
                "--hierarchy-path",
                HIERARCHY_PATH,
                "--output-path",
                "cerebellum_direction_vectors.nrrd",
            ],
        )

        assert result.exit_code == 0, result.output
        direction_vectors = VoxelData.load_nrrd("cerebellum_direction_vectors.nrrd")
        npt.assert_array_equal(direction_vectors.raw.shape, (12, 12, 12, 3))
        assert direction_vectors.raw.dtype == np.float32


def get_nan_interpolation_result(runner, nans_opt=None):
    args = [
        "--direction-vectors-path",
        "direction_vectors.nrrd",
        "--annotation-path",
        "annotation.nrrd",
        "--hierarchy-path",
        "hierarchy.json",
        "--metadata-path",
        "metadata.json",
        "--output-path",
        "interpolated_vectors.nrrd",
        "--restrict-to-hemisphere",
        "--restrict-to-layer",
    ]

    if nans_opt is not None:
        args.append(nans_opt)

    return runner.invoke(tested.interpolate, args)


def test_interpolate_nans():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Write input files to disk
        data = get_input_data()
        for name in ["annotation", "direction_vectors"]:
            VoxelData(data[name], voxel_dimensions=[25.0] * 3).save_nrrd(f"{name}.nrrd")
        for name in ["hierarchy", "metadata"]:
            with open(f"{name}.json", "w") as out:
                json.dump(data[name], out)

        # Run the CLI
        result = get_nan_interpolation_result(runner, "--nans")
        assert result.exit_code == 0

        # Check output
        direction_vectors = VoxelData.load_nrrd("interpolated_vectors.nrrd").raw
        region_mask = data["annotation"] > 0
        assert np.all(~np.isnan(direction_vectors[region_mask]))

        # Check exception
        del data["metadata"]["layers"]
        with open("metadata.json", "w") as out:
            json.dump(data["metadata"], out)

        result = get_nan_interpolation_result(runner, "--nans")
        assert "layers" in str(result.exception)

        result = get_nan_interpolation_result(runner)
        assert "--nans" in str(result.exception)


def get_interpolate_result(runner):
    return runner.invoke(
        tested.interpolate,
        [
            "--direction-vectors-path",
            "direction_vectors.nrrd",
            "--mask-path",
            "mask.nrrd",
            "--annotation-path",
            "annotation.nrrd",
            "--hierarchy-path",
            "hierarchy.json",
            "--metadata-path",
            "metadata.json",
            "--output-path",
            "interpolated_vectors.nrrd",
            "--restrict-to-hemisphere",
            "--restrict-to-layer",
        ],
    )


def test_interpolate_():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Write input files to disk
        data = get_input_data()
        for name in ["annotation", "direction_vectors"]:
            VoxelData(data[name], voxel_dimensions=[25.0] * 3).save_nrrd(f"{name}.nrrd")

        VoxelData(data["mask"].astype(np.uint8), voxel_dimensions=[25.0] * 3).save_nrrd("mask.nrrd")

        for name in ["hierarchy", "metadata"]:
            with open(f"{name}.json", "w") as out:
                json.dump(data[name], out)

        # Run the CLI
        result = get_interpolate_result(runner)
        assert result.exit_code == 0

        # Check exception (inconsistent voxel_dimensions)
        VoxelData(data["mask"].astype(np.uint8), voxel_dimensions=[10.0] * 3).save_nrrd("mask.nrrd")

        result = get_interpolate_result(runner)
        assert "voxel_dimensions" in str(result.exception)

"""
Generate direction vectors by means of Regiodesics,
https://bbpcode.epfl.ch/browse/code/viz/Regiodesics/tree/.

We first create a bottom surface and a top surface, referred
to as 'shells' by Regiodesics.

These shells are then passed to Regiodesics which generates
direction vectors flowing from the bottom to the top shell.

This algorithm is appropriate when the fibers of the brain region
follow streamlines which start from and end to well identified surfaces.
"""
from __future__ import annotations

import logging
from distutils.spawn import find_executable
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory

import numpy as np
from atlas_commons.typing import BoolArray, NDArray
from atlas_commons.utils import zero_to_nan
from voxcell import VoxelData  # type: ignore

from atlas_direction_vectors.algorithms.utils import compute_boundary
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


class RegiodesicsLabels:  # pylint: disable=too-few-public-methods
    """
    Class holding the voxel labels used by Regiodesics to identify voxels location.

    See https://bbpcode.epfl.ch/browse/code/viz/Regiodesics/tree/regiodesics/types.h.
    """

    INTERIOR = 1
    SHELL = 2
    BOTTOM = 3
    TOP = 4


def find_regiodesics_exec_or_raise(executable_name: str) -> str:
    """
    Find the specified Regiodesics executable. Raise if it wasn't found.

    Args:
        executable_name: name of the Regiodesics executable,
            e.g., 'layer_segmenter' or 'geodesics'.
    Returns:
        executable_path: a path to the quiered executable.
    Raises:
        FileExistsError if the executable cannot be found.
    """
    executable_path = find_executable(executable_name)
    if not executable_path:
        msg = (
            f"Regiodesics's {executable_name} was not found in this system.\n "
            f"Install Regiodesics or consider alternative algorithms."
        )
        raise FileExistsError(msg)

    return executable_path


def _popen_pipe_logging(tool: str, *args: str) -> None:
    """
    Logging for the execution of a specified tool.

    Args:
        tool: path to the executable to call.
        args: arguments specified for the execution
            of `tool`.
    """
    L.info("Calling command: %s, \nwith args: %s", tool, args)
    check_call([tool, *args])


def _get_border_mask(region: BoolArray) -> BoolArray:
    """
    Get the mask of the elements masked by `region` lying on the borders of the array.

    Args:
        region: 3D boolean array

    Returns:
        3D boolean array of the same shape as the input. It consists in a submask
        for the elements masked by `region` whichlying on the borders of the input array.

    """
    mask = np.zeros_like(region, dtype=bool)
    shape = mask.shape
    # pylint: disable=unsubscriptable-object
    mask[[0, shape[0] - 1], :, :] = True
    mask[:, [0, shape[1] - 1], :] = True
    mask[:, :, [0, shape[2] - 1]] = True
    return np.logical_and(mask, region)


def mark_with_regiodesics_labels(
    bottom: BoolArray, in_between: BoolArray, top: BoolArray
) -> NDArray[np.int8]:
    """Given 3 volumes, find the boundaries between them.

    The volume of interest `in_between` is supposed to be surrounded by the `bottom` and the `top`
    volumes, e.g, `bottom` is a lower layer and `bottom` is an upper layer. It may have a
    non-empty intersection with `bottom` and `top`.

    Args:
        bottom: boolean 3D mask of the bottom part.
        in_between: boolean 3D mask of the volume of interest.
        top: boolean 3D mask of the top part.

    Returns:
        marked(numpy.ndarray), a 3D array of RegiodesicsLabels marking the interior of
            `in_between` and its boundaries shared with `bottom` and `top`.

    Raises:
        AtlasDirectionVectorsError if the volume marked by `Bottom` or `Top` is empty.
    """
    shell = np.logical_or(
        compute_boundary(in_between, np.logical_not(in_between)),
        _get_border_mask(in_between),
    )

    def inner_boundary_with(mask: BoolArray):
        """
        Ensures that bottom and top boundaries lie in the computed shell.

        Args:
            mask: boolean 3D array holding the mask of the region whose boundary
                intersects with `shell`.
        Returns:
            boolean 3D numpy.ndarray array.
        """
        return np.logical_and(shell, np.logical_or(compute_boundary(in_between, mask), mask))

    marked = in_between * RegiodesicsLabels.INTERIOR
    marked[shell] = RegiodesicsLabels.SHELL
    marked[inner_boundary_with(bottom)] = RegiodesicsLabels.BOTTOM
    marked[inner_boundary_with(top)] = RegiodesicsLabels.TOP

    if np.count_nonzero(marked == RegiodesicsLabels.BOTTOM) == 0:
        raise AtlasDirectionVectorsError("Empty bottom volume is not supported by Regiodesics.")

    if np.count_nonzero(marked == RegiodesicsLabels.TOP) == 0:
        raise AtlasDirectionVectorsError("Empty top volume is not supported by Regiodesics.")

    return marked


def compute_direction_vectors(
    bottom: BoolArray, in_between: BoolArray, top: BoolArray
) -> NDArray[np.float32]:
    """
    Generate direction vectors for the `in_between` volume.

    The volume of interest `in_between` is assumed to share a boundary with
    a `bottom` and a `top` region.

    Args:
        bottom(numpy.ndarray): boolean 3D mask of the bottom part.
        in_between(numpy.ndarray): boolean 3D mask the volume of interest.
        top(numpy.ndarray): boolean 3D of the top part.

    Returns:
        np.float32 numpy.ndarray of shape (W, L, D, 3) holding a field of unit
        3D vectors over `in_between`. Voxels outside `in_between` are assigned a 3D vector
        with np.nan coordinates.
    """
    if np.count_nonzero(bottom) == 0 or np.count_nonzero(top) == 0:
        raise AtlasDirectionVectorsError(
            "The bottom or the top part of the region is missing.\n"
            "Regiodesics cannot handle this incomplete input."
        )

    marked = mark_with_regiodesics_labels(bottom, in_between, top)
    regiodesics_path = find_regiodesics_exec_or_raise("direction_vectors")

    with TemporaryDirectory() as temp_dir:
        dummy_voxel_sizes = (1.0, 1.0, 1.0)
        VoxelData(marked.astype(np.int8), dummy_voxel_sizes).save_nrrd(
            str(Path(temp_dir, "marked_int8.nrrd")), encoding="raw"
        )
        _popen_pipe_logging(
            regiodesics_path,
            "-s",
            str(Path(temp_dir, "marked_int8.nrrd")),
            "-o",
            str(Path(temp_dir, "direction_vectors.nrrd")),
        )
        direction_vectors = VoxelData.load_nrrd(Path(temp_dir, "direction_vectors.nrrd")).raw
        direction_vectors = direction_vectors.astype(np.float32)
        zero_to_nan(direction_vectors)
        return direction_vectors

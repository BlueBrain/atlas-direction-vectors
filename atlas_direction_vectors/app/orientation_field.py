"""Turn direction vectors into quaternions interpreted as 3D orientations

This command takes as input a field of 3D unit vectors and outputs a field of quaternions
under the format (w, x, y, z) defined over the annotated volume.

If v is a 3D direction vector, the corresponding quaternion is defined by the formula
``q = e cross v + (e dot v + |e||v|)`` where

- ``e = (0, 1, 0)``,
- ``|.|`` denotes the 3D Euclidean norm,
- ``cross`` denotes the cross product,
- ``dot`` denotes the scalar product.

In particular q maps e to v, i.e., ``q e q^{-1} = v``.
If ``q`` is interpreted as an orientation, i.e., a 3D orthonormal frame, this implies that
the y-axis of q is directed along e. The latter is a requirement of the morphology orientation
convention used by the placement algorithm, see
https://bbpteam.epfl.ch/documentation/projects/placement-algorithm/latest/index.html.
"""
import logging

import click
import voxcell  # type: ignore
from atlas_commons.app_utils import EXISTING_FILE_PATH, log_args

from atlas_direction_vectors.algorithms.utils import vector_to_quaternion

L = logging.getLogger(__name__)


@click.command()
@click.option(
    "--direction-vectors-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to an nrrd file containing a field of 3D unit vectors defined over a 3D volume."),
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Path to the file where voxel orientations will be saved as a field of 4D vectors"
    " (w, x, y, z) over a 3D volume. A vector (w, x, y, z) represents a quaternion with "
    "imaginary part (x, y, z). The nrrd data type is the same as the input type."
    "NaN vectors indicate out-of-domain voxels but also voxels for which an orientation could"
    " not be derived.",
)
@log_args(L)
def cmd(
    direction_vectors_path: str,
    output_path: str,
) -> None:
    """Turn direction vectors into quaternions interpreted as 3D orientations."""

    direction_vectors = voxcell.VoxelData.load_nrrd(direction_vectors_path)
    quaternions = vector_to_quaternion(direction_vectors.raw)
    voxcell.OrientationField(
        quaternions, direction_vectors.voxel_dimensions, direction_vectors.offset
    ).save_nrrd(output_path)

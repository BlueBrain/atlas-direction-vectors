"""A library to transform vector fields.

A N-dimensional vector field assigns a N-dimensional vector to each point of a domain.
A 1-dimensional vector field is a scalar field, e.g., a heat field or a height field.

Vector fields can be represented by numpy arrays. The last dimension in the shape of the array is
N, i.e., shape = (..., N). The other dimensions are those of the domain on which the vector field is
defined.

In the context of the atlas, the volumetric files can all be interpreted as vector fields over
a 3D domain, the 3D space enclosing the brain. An example of a 3D vector field is the field of
fiber tracts direction vectors, also simply called the direction vector field. It can be represented
by a numpy array of shape (W, H, D, 3). An example of a 4D vector field is the orientation field of
a brain region, i.e., an assignment of a quaternion to each voxel of the 3D region. This can be
represented by a numpy array of shape (W, H, D, 4).

Note: this code duplicates nse/atlas-analysis/atlas_analysis/vector_field.py.
This duplication is done on purpose so as to remove the dependency on atlas-analysis.
See https://bbpteam.epfl.ch/project/issues/browse/NSETM-1454.
"""

from typing import Optional

import numpy as np
from atlas_commons.typing import BoolArray, FloatArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError

INTERPOLATION_ALGORITHMS = {
    "linear": LinearNDInterpolator,
    "nearest-neighbour": NearestNDInterpolator,
}


def interpolate(
    field: FloatArray,
    unknown_values_mask: BoolArray,
    known_values_mask: Optional[BoolArray] = None,
    interpolator: str = "linear",
) -> None:
    """
    Interpolate in-place a vector field.

    The underlying algorithms are provided by scipy:
    https://docs.scipy.org/doc/scipy/reference/interpolate.html

    Note: Values located on the boundary of the input array will not be interpolated
    by LinearNDInterpolator.

    Args:
        field(numpy.ndarray): numeric array of shape (..., N) to be interpolated.
            This is a field of N-dimensional vectors defined on a volume of arbitrary dimensions.
        unknown_values_mask(numpy.ndarray): boolean array of shape `field.shape[:-1]`. A mask
            for the unknown values, i.e., a mask for the locations where `field` should be
            interpolated.
        known_values_mask(numpy.ndarray): (Optional) boolean array of shape `field.shape[:-1]`.
            A mask for the known values. Defaults to None. In this case the known values are all
            values except the `unknown_values`.
        interpolator(str): the interpolator algorithm. Either 'linear' or 'nearest-neighbour'. The
            corresponding implementations are scipy's LinearNDInterpolator and
            NearestNDInterpolator.
    """
    if interpolator not in INTERPOLATION_ALGORITHMS:
        raise AtlasDirectionVectorsError(
            f"Unknown interpolator. `interpolator` must be set with one "
            f"of the following {INTERPOLATION_ALGORITHMS.keys()}"
        )

    interpolator_ = INTERPOLATION_ALGORITHMS[interpolator]
    if known_values_mask is None:
        known_values_mask = ~unknown_values_mask
    known_indices = np.where(known_values_mask)
    unknown_indices = np.where(unknown_values_mask)
    interpolated_values = interpolator_(known_indices, field[known_values_mask])(unknown_indices)
    field[unknown_values_mask] = interpolated_values

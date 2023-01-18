"""Low-level tools for the computation of direction vectors"""

import numpy as np  # type: ignore
from atlas_commons.utils import FloatArray, NumericArray, normalize
from scipy.ndimage import gaussian_filter  # type: ignore
from scipy.ndimage import generate_binary_structure  # type: ignore
from scipy.signal import correlate  # type: ignore


def compute_blur_gradient(scalar_field: FloatArray, gaussian_stddev=3.0) -> FloatArray:
    """
    Blurs a scalar field and returns its normalized gradient.

    A Gaussian filter (blur) with standard deviation `gaussian_stdev`
    is applied to the input field. The function returns the normalized
    gradient of the filtered field.

    Arguments:
        scalar_field: floating point scalar field defined over
            a 3D volume.
        gaussian_stddev: standard deviation of the Gaussian kernel used by the
            Gaussian filter.
    Returns:
        numpy.ndarray of float type. A 3D unit vector field over the underlying 3D volume
        of the input scalar field. This vector contains np.nan vectors if the normalization
        process encounters some zero vectors.
    Raises:
        ValueError if the input field is not of floating point type.
    """
    if not np.issubdtype(scalar_field.dtype, np.floating):
        raise ValueError(
            f"The input field must be of floating point type. Got {scalar_field.dtype}."
        )
    blurred = gaussian_filter(scalar_field, sigma=gaussian_stddev)
    gradient = np.array(np.gradient(blurred))
    gradient = np.moveaxis(gradient, 0, -1)
    normalize(gradient)
    return gradient


def _quaternion_from_vectors(  # pylint: disable=invalid-name
    s: NumericArray, t: NumericArray
) -> NumericArray:
    """
    Returns the quaternion (s cross t) + (s dot t + |s||t|).

    This quaternion q maps s to t, i.e., qsq^{-1} = t.

    Args:
        s: numeric array of shape (3,) or (N, 3).
        t: numeric array of shape (N, 3) if s has two dimensions and its first dimension is N.
    Returns:
        Numeric array of shape (N, 4) where N is the first dimension of t.
        This data is interpreted as a 1D array of quaternions with size N. A quaternion is a 4D
        vector [w, x, y, z] where [x, y, z] is the imaginary part.
    """
    w = np.matmul(s, np.array(t).T) + np.linalg.norm(s, axis=-1) * np.linalg.norm(t, axis=-1)
    return np.hstack([w[:, np.newaxis], np.cross(s, t)])


def vector_to_quaternion(vector_field: FloatArray) -> FloatArray:
    """
    Find quaternions which rotate [0.0, 1.0, 0.0] to each vector in `vector_field`.

    A returned quaternion is of the form [w, x, y, z] where [x, y, z] is imaginary part and w the
     real part.
    The specific choice of returned quaternion is documented in _quaternion_from_vectors.

    Arguments:
        vector_field: field of floating point 3D unit vectors, i.e., a float array of shape
            (..., 3).

    Returns:
        numpy.ndarray of shape (..., 4) and of the same type as the input
        vector field.

    """
    if not np.issubdtype(vector_field.dtype, np.floating):
        raise ValueError(
            f"The input field must be of floating point type. Got {vector_field.dtype}."
        )
    quaternions = np.full(vector_field.shape[:-1] + (4,), np.nan, dtype=vector_field.dtype)
    non_nan_mask = ~np.isnan(np.linalg.norm(vector_field, axis=-1))
    quaternions[non_nan_mask] = _quaternion_from_vectors(
        [0.0, 1.0, 0.0], vector_field[non_nan_mask]
    )
    return quaternions


def compute_boundary(v_1, v_2):
    """Compute the boundary shared by two volumes.

    The voxels of `v_1` (resp. of `v_2`) are labeled with the value 1 (resp. 8).
    We build the filter corresponding to the 6 neighbour voxels that share a face
    with a reference voxel. We apply a covolution of the filter with the labeled volume.
    In the resulting labeled volume, the `v_1` voxels with label > 8 are exactly those voxels
    that share a face with at least one voxel of `v_2`.
    (The interior voxels of `v_1` have labels bounded above by 7).

    Check https://docs.scipy.org/doc/scipy/reference/ndimage.html for the doc
    of the functions generate_binary_structure and correlate used below.

    Args:
        v_1(numpy.ndarray): boolean 3D array holding the mask of the first volume.
        v_2(numpy.ndarray): boolean 3D array holding the mask of the second volume.

    Returns:
        shared_boundary(numpy.ndarray), 3D boolean array holding the mask of the boundary shared
        by `v_1` and `v_2`. This corresponds to a subset of `v_1`.
    """

    filter_ = generate_binary_structure(3, 1).astype(int)
    full_volume = correlate(v_1 * 1 + v_2 * 8, filter_, mode="same")

    return np.logical_and(v_1, full_volume > 8)

"""
Utility functions for direction vectors computations.
"""
import logging
import warnings
from pathlib import Path

import numpy as np
from atlas_commons.typing import BoolArray, FloatArray

L = logging.getLogger(__name__)
logging.captureWarnings(True)

TEST_PATH = Path(Path(__file__).parent.parent)
HIERARCHY_PATH = str(Path(TEST_PATH, "1.json"))


def warn_on_nan_vectors(
    direction_vectors: FloatArray, region_mask: BoolArray, region_name: str
) -> None:
    """
    Warn on the consequences of (NaN, NaN, NaN) vectors when some are reported.

    Args:
        direction_vectors: float array of shape (W, H, D, 3) holding the direction vectors
            of the region `region_name`. Voxels outside of `region_mask` are assigned the value
            [np.nan, np.nan, np.nan].
        region_mask: boolean 3D mask of the voxels of the 3D region `region_name`.
        region_name: name of the region of interest, e.g., "Isocortex", "Thalamus", "Cerebellum",
            etc.
    """
    nan_mask = np.isnan(direction_vectors[region_mask])
    if not np.any(nan_mask):  # This would cause np.mean to raise a warning
        return
    nans = np.mean(nan_mask)
    if nans > 0.0:
        warnings.warn(
            f"(NaN, NaN, NaN) direction vectors in {float(nans):.3%} of the {region_name} voxels",
            UserWarning,
        )
        if region_name.lower() == "isocortex":
            warnings.warn(
                "(NaN, NaN, NaN) direction vectors are likely to prevent you from "
                "splitting layer 2/3.",
                UserWarning,
            )
        warnings.warn(
            "Such vectors are likely to produce problematic placement hints. "
            "Consider interpolating (NaN, NaN, NaN) vectors by valid ones. "
            "See ``atlas-direction-vectors direction-vectors interpolate --help``.",
            UserWarning,
        )

from distutils.spawn import find_executable

import pytest

skip_if_no_regiodesics = pytest.mark.skipif(
    find_executable("direction_vectors") is None,
    reason="direction_vectors from regiodesics is not present",
)

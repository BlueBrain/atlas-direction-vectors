"""version"""
from pkg_resources import get_distribution  # type: ignore

VERSION = get_distribution("atlas_direction_vectors").version

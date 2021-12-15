"""The atlas-direction-vectors command line launcher"""

import logging

import click

from atlas_direction_vectors.app import direction_vectors, orientation_field
from atlas_direction_vectors.version import VERSION

L = logging.getLogger(__name__)


def cli():
    """The main CLI entry point"""
    logging.basicConfig(level=logging.INFO)
    group = {
        "direction-vectors": direction_vectors.app,
        "orientation-field": orientation_field.cmd,
    }
    help_str = "The main CLI entry point."

    app = click.Group("atlas_direction_vectors", group, help=help_str)
    app = click.version_option(VERSION)(app)
    app()

"""Generate and save the direction vectors for different regions of the mouse brain"""

# pylint: disable=import-outside-toplevel,too-many-arguments
import json
import logging
from copy import copy
from functools import partial

import click  # type: ignore
import numpy as np
import voxcell  # type: ignore
from atlas_commons.app_utils import (
    EXISTING_FILE_PATH,
    assert_meta_properties,
    common_atlas_options,
    log_args,
    set_verbose,
)
from atlas_commons.utils import compute_halfway
from joblib import Parallel, delayed

from atlas_direction_vectors import cerebellum as cerebellum_
from atlas_direction_vectors import thalamus as thalamus_
from atlas_direction_vectors.algorithms import (
    direction_vectors_from_center,
    layer_based_direction_vectors,
)
from atlas_direction_vectors.algorithms.layer_based_direction_vectors import (
    compute_layered_region_direction_vectors,
)
from atlas_direction_vectors.algorithms.regiodesics import find_regiodesics_exec_or_raise
from atlas_direction_vectors.exceptions import AtlasDirectionVectorsError
from atlas_direction_vectors.interpolation import interpolate_vectors
from atlas_direction_vectors.isocortex import ISOCORTEX_ALGORITHMS

L = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def app(verbose):
    """Run the different direction vectors CLI.

    Direction vectors are 3D unit vectors associated to voxels of a brain region.
    They represent the directions of the fiber tracts and their streamlines are assumed
    to cross tranversely layers in laminar brain regions.

    Direction vectors are used in placement-algorithm to set cells orientations.

    Direction vectors are also used to compute placement hints (see the placement_hints module)
    and split layer 2/3 of the AIBS mouse isocortex.
    """
    set_verbose(L, verbose)


@app.command()
@common_atlas_options
@click.option(
    "--output-path",
    required=True,
    help="Path of file to write the direction vectors to.",
)
@log_args(L)
def cerebellum(annotation_path, hierarchy_path, output_path):
    """Generate and save the direction vectors of the AIBS mouse cerebellar cortex.

    This command relies on the computation of the gradient of a Gaussian blur
    applied to specific parts of the cerebellar cortex.

    The output file is an nrrd file enclosing a float32 array of shape (W, H, D, 3)
    where (W, H, D) is the shape the input annotation array.

    The vector [nan, nan, nan] is assigned to any voxel outside the cerebellar cortex.

    """
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    dir_vectors = cerebellum_.compute_direction_vectors(region_map, annotation)
    annotation.with_data(dir_vectors).save_nrrd(output_path)


@app.command()
@common_atlas_options
@click.option("--output-path", required=True, help="Path of file to write.")
@click.option(
    "--algorithm",
    default="regiodesics",
    type=click.Choice(list(ISOCORTEX_ALGORITHMS.keys())),
    required=False,
    help="Algorithm to use for the computation of the direction vector field. "
    "Defaults to 'regiodesics'.",
)
@log_args(L)
def isocortex(annotation_path, hierarchy_path, output_path, algorithm):
    """Generate and save the direction vectors of the mouse isocortex.

    This command relies on Regiodesics.

    The output file is an nrrd file enclosing a float32 array of shape (W, H, D, 3)
    where (W, H, D) is the shape the input annotation array.

    The vector [nan, nan, nan] is assigned to any voxel out of the isocortex.
    The annotation file can enclose the isocortex or a superset.
    """
    from atlas_direction_vectors.isocortex import compute_direction_vectors

    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    if algorithm == "regiodesics":
        find_regiodesics_exec_or_raise("direction_vectors")
    dir_vectors = compute_direction_vectors(region_map, annotation, algorithm)
    annotation.with_data(dir_vectors).save_nrrd(output_path)


@app.command()
@common_atlas_options
@click.option("--output-path", required=True, help="Path of file to write.")
@log_args(L)
def thalamus(annotation_path, hierarchy_path, output_path):
    """Generate and save the direction vectors of the mouse thalamus.

    This command relies on the computation of the gradient of a Gaussian blur
    applied to the reticular nucleus of the thalamus and its complement inside
    the thalamus.

    The output file is an nrrd file enclosing a float32 array of shape (W, H, D, 3)
    where (W, H, D) is the shape the input annotation array.

    The vector [nan, nan, nan] is assigned to any voxel out of the thalamus.
    The annotation file can contain the thalamus or a superset.
    For the algorithm to work properly, some space should separate the boundary
    of the thalamus from the boundary of its enclosing array.
    """
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    dir_vectors = thalamus_.compute_direction_vectors(region_map, annotation)
    annotation.with_data(dir_vectors).save_nrrd(output_path)


METADATA_HELP_STR = (
    "Path to the json metadata file of the brain region of interest. It must enclose the "
    "definition of the region and also the definition of the region layers if "
    "--restrict-to-layer is specified. Regions are defined through regular expressions"
    " which can be consumed by voxcell.RegionMap.find. "
    "See atlas-direction-vectors/atlas_direction_vectors/data/metadata for examples."
)
ANNOTATION_HELP_STR = (
    "Path to the annotation nrrd file. It can contain the whole annotated brain volume or a "
    "superset of the region defined by the 'region' section of the metadata json file."
)
RESTRICT_TO_HEMISPHERE_HELP_STR = (
    "Performs interpolation in each hemisphere separately. The region of interest is assumed "
    "to be symmetric wrt to z = half_z_dimensions."
)
RESTRICT_TO_LAYER_HELP_STR = (
    "Performs interpolation in each layer separately. The region of interest is assumed "
    "to be made of layers. These layers must be listed in the metadata json file. See "
    "--metadata-path."
)


@app.command()
@click.option(
    "--direction-vectors-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to the nrrd files containing direction vectors."),
)
@click.option(
    "--nans",
    is_flag=True,
    required=False,
    help=(
        "Interpolate each [NaN, NaN, NaN] direction vector of the voxels in the region of interest"
        "by non-NaN direction vectors of nearby voxels. Must be set if --mask-path is "
        "not specified."
    ),
    default=False,
)
@click.option(
    "--mask-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=(
        "Path to the nrrd files containing a mask of the voxels whose direction vectors will be"
        " interpolated. A non-zero values are used to filter voxels in."
    ),
    default=None,
)
@common_atlas_options
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=METADATA_HELP_STR,
)
@click.option(
    "--output-path",
    required=True,
    help="Path of the file where to write the modified direction vectors.",
)
@click.option(
    "--restrict-to-hemisphere",
    is_flag=True,
    help=RESTRICT_TO_HEMISPHERE_HELP_STR,
    default=False,
)
@click.option(
    "--restrict-to-layer",
    is_flag=True,
    help=RESTRICT_TO_LAYER_HELP_STR,
    default=False,
)
@log_args(L)
def interpolate(  # pylint: disable=too-many-arguments,too-many-locals
    direction_vectors_path,
    annotation_path,
    nans,
    mask_path,
    hierarchy_path,
    metadata_path,
    output_path,
    restrict_to_hemisphere,
    restrict_to_layer,
):
    """
    Interpolate the direction vectors of the voxels in the mask by those of voxels out of the
    mask.

    The direction vector of each voxel in the mask is interpolated by non-NaN direction
    vectors of nearby voxels out of the mask if --mask-path is specified.

    The NaN direction vectors are interpolated by non-NaN direction vectors of neighboring voxels
    if --nans is specified. In this case, each [NaN, NaN, NaN] direction vector of a voxel in the
    region of interest is interpolated by non-NaN direction vectors of nearby voxels.

    Interpolation is restricted to voxels inside the region of interest which is specified in the
    json metadata file.

    The interpolation is made separately on each hemisphere and on each layer if the flags
    --restrict-to-hemisphere and --restrict-to-layer are specified.

    Note: When direction vectors are created, the voxels lying outside of the region of interest are
    assigned [NaN, NaN, NaN] direction vectors. These vectors should not be interpolated. However,
    some voxels inside the region of interest can bear NaN direction vectors because of the
    algorithm limitations. This function aims at finding a sensible value for those vectors.
    """

    if mask_path is None and not nans:
        raise AtlasDirectionVectorsError(
            "None of --mask-path or --nans was specified. You must specify at least one of these "
            "two options."
        )

    direction_vectors = voxcell.VoxelData.load_nrrd(direction_vectors_path)
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    voxel_data = [annotation, direction_vectors]

    mask = None
    if mask_path is not None:
        mask = voxcell.VoxelData.load_nrrd(mask_path)
        voxel_data.append(mask)

    # Check nrrd metadata consistency
    assert_meta_properties(voxel_data)

    with open(metadata_path, "r", encoding="utf-8") as file_:
        metadata = json.load(file_)

    if restrict_to_layer and "layers" not in metadata.keys():
        raise AtlasDirectionVectorsError(
            "The 'layers' key is required in metadata when --restrict-to-layer is specified. "
            "Please provide the definition of the layers in your json metadata file."
        )

    region_map = voxcell.RegionMap.load_json(hierarchy_path)

    interpolate_vectors(
        annotation.raw,
        region_map,
        metadata,
        direction_vectors.raw,
        nans,
        mask.raw != 0 if mask is not None else None,
        restrict_to_hemisphere,
        restrict_to_layer,
    )

    direction_vectors.save_nrrd(output_path)


@app.command()
@common_atlas_options
@click.option(
    "--output-path",
    required=True,
    help="Path of file to write the direction vectors to.",
)
@click.option("--outside-brain", type=int)
@click.option("--layer", type=(str, int), multiple=True)
@click.option("--hemisphere/--no-hemisphere", default=True)
@log_args(L)
def layer_region(annotation_path, hierarchy_path, output_path, outside_brain, layer, hemisphere):
    """Generate and save the direction vectors for an arbitrary region

    The --layer options is ordered with the value attached to the specific voxels:

        --outside-brain  10
        --layer VISpm1     2
        --layer VISpm2/3   1
        --layer VISpm4     0
        --layer VISpm5    -1
        --layer VISpm6a   -2
        --layer VISpm6b   -3

    """
    from atlas_direction_vectors.region import layered_region

    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    dir_vectors = layered_region(annotation, region_map, outside_brain, dict(layer), hemisphere)
    annotation.with_data(dir_vectors).save_nrrd(output_path)


@app.command()
@common_atlas_options
@click.option(
    "--output-path",
    required=True,
    help="Path of file to write the direction vectors to.",
)
@click.option("--source", type=str, help="atlas acronym to be used as the source")
@click.option("--region", type=str, help="atlas acronym region of interest")
@click.option("--target", type=str, help="atlas acronym to be used as the target")
@click.option("--algorithm", type=click.Choice(list(layer_based_direction_vectors.ALGORITHMS)))
@click.option(
    "--hemisphere-option",
    type=click.Choice(
        [h.name.lower() for h in layer_based_direction_vectors.HemisphereOppositeOption]
    ),
    default=layer_based_direction_vectors.HemisphereOppositeOption.NO_SPLIT,
)
@log_args(L)
def source_target_layered_region(
    annotation_path,
    hierarchy_path,
    output_path,
    source,
    region,
    target,
    algorithm,
    hemisphere_option,
):
    """Generate and save the direction vectors for an arbitrary region
    --source '@.*6[b]$'
    --target "@.*1$"
    --region VISpm
    """
    from atlas_direction_vectors.region import source_target_layered_region as stlr

    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    hemisphere_option = layer_based_direction_vectors.HemisphereOppositeOption[
        hemisphere_option.upper()
    ]

    dir_vectors = stlr(
        annotation,
        region_map,
        algorithm,
        source,
        region,
        target,
        hemisphere_option,
    )
    annotation.with_data(dir_vectors).save_nrrd(output_path)


@app.command()
@common_atlas_options
@click.option(
    "--output-path",
    required=True,
    help="Path of file to write the direction vectors to.",
)
@click.option(
    "--region", type=str, required=True, help="atlas acronym for region of the direction vectors"
)
@click.option(
    "--center", type=(float, float, float), default=None, help="example: --center 148.6 79.8 113.9"
)
@log_args(L)
def from_center(annotation_path, hierarchy_path, output_path, region, center):
    """Generate and save the direction vectors for an arbitrary region"""
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)

    dir_vectors = direction_vectors_from_center.command(region_map, annotation.raw, region, center)
    annotation.with_data(dir_vectors).save_nrrd(output_path)


def _create_subregion_direction_vectors(region_id, region_config, region_map, annotation):
    """Create direciton vectors for a subregion, to be used in parallel."""
    acronym = region_map.get(region_id, "acronym")
    metadata = copy(region_config["metadata"])

    for i, query in enumerate(metadata["layers"]["queries"]):
        if "***" in query:
            metadata["layers"]["queries"][i] = query.replace("***", acronym)
        metadata["region"]["query"] = metadata["region"]["query"].replace("***", acronym)

    # names are not needed, but mandatory from atlas-commons
    metadata["layers"]["names"] = metadata["layers"]["queries"]

    region_to_weight = {}
    for key, weight in region_config["region_to_weight"].items():
        if "***" in key:
            key = key.replace("***", acronym)
        region_to_weight[key] = weight

    # find if the region is split in hemispheres (no voxel near the halfplane)
    z_halfway = compute_halfway(annotation.shape[2])
    annot = np.zeros_like(annotation.raw, dtype=int)
    subregion_ids = list(region_map.find(region_id, "id", with_descendants=True))
    annot[np.isin(annotation.raw, subregion_ids)] = 1

    has_hemisphere = not any(annot[:, :, z_halfway - 1 : z_halfway + 1].flatten())
    L.info("Subregion %s has hemispheres: %s", region_map.get(region_id, "name"), has_hemisphere)

    return compute_layered_region_direction_vectors(
        region_map=region_map,
        annotation=annotation,
        metadata=metadata,
        region_to_weight=region_to_weight,
        shading_width=region_config.get("shading_width", 4),
        expansion_width=region_config.get("expansion_width", 8),
        has_hemispheres=has_hemisphere,
    )


def _get_region_ids(region, region_map, region_config, annotation):
    """Fetch all the region ids that need to be processed in parallel."""
    if not region_config["region_query"].get("with_descendants", False):

        return region_map.find(
            region_config["region_query"]["query"], region_config["region_query"]["attribute"]
        )

    subregion_ids = []
    all_ids = np.unique(annotation.raw)
    for subregion_id in region_map.find(
        region_config["region_query"]["query"],
        region_config["region_query"]["attribute"],
        with_descendants=True,
    ):
        parent_id = region_map.get(subregion_id, "parent_structure_id")
        if (
            subregion_id in all_ids
            and region_map.is_leaf_id(subregion_id)
            and parent_id not in subregion_ids
        ):
            subregion_ids.append(parent_id)
    return subregion_ids


DEFAULT_CONFIG = {
    "cerebellum": {
        "region_query": {
            "query": "Cerebellar cortex",
            "attribute": "name",
            "with_descendants": True,
        },
        "metadata": {
            "region": {
                "name": "Cerebellar cortex",
                "query": "@^***|cbf$",
                "attribute": "acronym",
                "with_descendants": True,
            },
            "layers": {
                "queries": ["***mo", "***pu", "***gr", "cbf"],
                "attribute": "acronym",
                "with_descendants": True,
            },
        },
        "region_to_weight": {
            "***mo": 1.0,
            "***pu": 0.0,
            "***gr": -1.0,
            "cbf": -5.0,
            "outside_of_brain": 3.0,
        },
    },
    "isocortex": {
        "region_query": {"query": "Isocortex", "attribute": "name", "with_descendants": True},
        "metadata": {
            "region": {
                "name": "Extended Isocortex",
                "query": "@^***|lfbs$",
                "attribute": "acronym",
                "with_descendants": True,
            },
            "layers": {
                "queries": [
                    "@***1[ab]?$",
                    "@***[2-3][ab]?$",
                    "@***4[ab]?$",
                    "@***5[ab]?$",
                    "@***6[ab]?$",
                    "lfbs",
                ],
                "attribute": "acronym",
                "with_descendants": True,
            },
        },
        "region_to_weight": {
            "@***1[ab]?$": 6,
            "@***[2-3][ab]?$": 5,
            "@***4[ab]?$": 3,
            "@***5[ab]?$": 2,
            "@***6[ab]?$": 1,
            "lfbs": -2,
            "outside_of_brain": 0,
        },
    },
    "main olfactory bulb": {
        "region_query": {
            "query": "Main Olfactory bulb",
            "attribute": "name",
            "with_descendants": False,
        },
        "metadata": {
            "region": {
                "name": "Main olfactory bulb",
                "query": "MOB",
                "attribute": "acronym",
                "with_descendants": True,
            },
            "layers": {
                "queries": [
                    "MOBgl",
                    "MOBopl",
                    "MOBmi",
                    "MOBipl",
                    "MOBgr",
                ],
                "attribute": "acronym",
                "with_descendants": True,
            },
        },
        "region_to_weight": {
            "MOBgl": 5,
            "MOBopl": 4,
            "MOBmi": 3,
            "MOBipl": 2,
            "MOBgr": 1,
            "outside_of_brain": 0,
        },
    },
    "thalamus": {
        "region_query": {
            "query": "Thalamus",
            "attribute": "name",
            "with_descendants": False,
        },
        "metadata": {
            "region": {
                "name": "Thalamus",
                "query": "TH",
                "attribute": "acronym",
                "with_descendants": True,
            },
            "layers": {
                "queries": [
                    "@^AD$|^AMd$|^AMv$|^AV$|^CL$|^CM$|^Eth$|^IAD$|^IAM$|^IGL$|^IMD$|^IntG$|^LD$|^LGd-co$|^LGd-ip$|^LGd-sh$|^LGv_O$|^LP$|^MD_O$|^MGd$|^MGm$|^MGv$|^PCN$|^PF$|^PIL$|^PO$|^POL$|^PR$|^PT$|^PVT$|^PoT$|^RE$|^RH$|^SGN$|^SMT$|^SPA$|^SPFm$|^SPFp$|^SubG$|^TH_O$|^VAL$|^VM$|^VPL$|^VPLpc$|^VPM$|^VPMpc$|^Xi$",
                    "RT",
                ],
                "attribute": "acronym",
                "with_descendants": True,
            },
        },
        "region_to_weight": {
            "@^AD$|^AMd$|^AMv$|^AV$|^CL$|^CM$|^Eth$|^IAD$|^IAM$|^IGL$|^IMD$|^IntG$|^LD$|^LGd-co$|^LGd-ip$|^LGd-sh$|^LGv_O$|^LP$|^MD_O$|^MGd$|^MGm$|^MGv$|^PCN$|^PF$|^PIL$|^PO$|^POL$|^PR$|^PT$|^PVT$|^PoT$|^RE$|^RH$|^SGN$|^SMT$|^SPA$|^SPFm$|^SPFp$|^SubG$|^TH_O$|^VAL$|^VM$|^VPL$|^VPLpc$|^VPM$|^VPMpc$|^Xi$": 1.0,
            "RT": 0.0,
            "outside_of_brain": 3.0,
        },
    },
}


@app.command()
@common_atlas_options
@click.option(
    "--output-path",
    required=True,
    help="Path of file to write the direction vectors to.",
)
@click.option("--config-path", type=str, required=False, default=None, help="path to config file")
@log_args(L)
def from_config(annotation_path, hierarchy_path, output_path, config_path):
    """General function to compute direction vector from a configuration file."""
    if config_path is None:
        config = DEFAULT_CONFIG
    else:
        with open(config_path) as config_file:
            config = json.load(config_file)

    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    direction_vectors = np.full(annotation.raw.shape + (3,), np.nan, dtype=np.float32)

    for region, region_config in config.items():
        region_ids = _get_region_ids(region, region_map, region_config, annotation)
        L.info("Computing direction vectors for %s with %s subregions.", region, len(region_ids))
        f = partial(
            _create_subregion_direction_vectors,
            region_config=region_config,
            region_map=region_map,
            annotation=annotation,
        )
        with Parallel(n_jobs=20, verbose=10) as parallel:
            for subregion_direction_vectors in parallel(
                delayed(f)(region_id) for region_id in region_ids
            ):
                # Assembles subregion direction vectors.
                subregion_mask = np.logical_not(np.isnan(subregion_direction_vectors))
                direction_vectors[subregion_mask] = subregion_direction_vectors[subregion_mask]

        annotation.with_data(direction_vectors).save_nrrd(output_path)

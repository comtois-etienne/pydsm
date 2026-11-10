"""Backward-compatible imports for the split command modules."""

import os

import imageio.v3 as iio
import pycityjson.io as cjio
import time

import pydsm.geo as geo
import pydsm.utils as utils

MUTE_SPEECH = True


def _print(string: str, end: str = '\n') -> None:
    """Print a command message and optionally speak its meaningful text."""
    print(string, end=end)
    splitted = string.split("'")
    if len(splitted) > 1:
        string = ''.join(
            item if item.startswith(' ') or item.endswith(' ')
            else utils.get_filename(item)
            for item in splitted
        )
    spoken = ''
    for item in string.split(' '):
        if item.startswith(('http', 'www')):
            spoken += 'the displayed link '
        elif item.isupper():
            spoken += ' '.join(item) + ' '
        else:
            spoken += item + ' '
    if not MUTE_SPEECH:
        os.system(f'say "{spoken}"')

from pydsm.geo_cmd import (
    compute_ndsm,
    display_epsg,
    display_geotiff_info,
    geotiff_registration,
    merge_geotiffs,
    overlay_values,
    reproject_geotiff,
    resize_geotiff_like,
    rescale_geotiff,
    to_cmap,
    to_xyz,
    translate_geotiff,
)
from pydsm.rbc_cmd import (
    generate_cityjson_terrain,
    generate_rbc_cityjson_tiles,
    generate_rbc_wavefront,
    generate_rbc_wavefront_tiles,
    generate_terrain_cityjson_tiles,
)
from pydsm.sat_cmd import (
    _download_cityjson_tile,
    _tile_corners,
    _tile_origins,
)
from pydsm.shp_cmd import crop_geotiff, csv_to_shapefile, extract_by_zone
from pydsm.tile_cmd import (
    export_npz_tile,
    export_npz_tiles,
    export_png_tile,
    export_png_tiles,
    extract_tiles,
)
from pydsm import sat_cmd, tile_cmd


def compute_ndsm_dir(*args, **kwargs):
    """Compatibility wrapper for :func:`pydsm.tile_cmd.compute_ndsm_dir`."""
    tile_cmd.geo = geo
    tile_cmd.iio = iio
    tile_cmd.compute_ndsm = compute_ndsm
    return tile_cmd.compute_ndsm_dir(*args, **kwargs)


def load_tile_in_cityjson(*args, **kwargs):
    """Compatibility wrapper for :func:`pydsm.sat_cmd.load_tile_in_cityjson`."""
    sat_cmd.cjio = cjio
    sat_cmd.time = time
    sat_cmd._download_cityjson_tile = _download_cityjson_tile
    return sat_cmd.load_tile_in_cityjson(*args, **kwargs)


def _download_with_retry(*args, **kwargs):
    """Compatibility wrapper for :func:`pydsm.sat_cmd._download_with_retry`."""
    sat_cmd.time = time
    sat_cmd._print = _print
    return sat_cmd._download_with_retry(*args, **kwargs)

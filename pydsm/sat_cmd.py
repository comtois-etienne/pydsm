"""
Commands for processing geotiff files
"""


import os
import time
from datetime import date
from typing import Callable

import numpy as np
import pycityjson.io as cjio
from osgeo import gdal

import pydsm.geo as geo
import pydsm.sat as sat
from pydsm import const
from pydsm.cmd import _print

def _download_with_retry(
    loader: Callable[..., gdal.Dataset],
    label: str,
    verbose: bool,
    max_attempts: int = 4,
    **kwargs: object,
) -> gdal.Dataset:
    """Download a dataset with exponential backoff after failures.

    :param loader: dataset download function.
    :param label: dataset label used in status messages.
    :param verbose: whether to print retry messages.
    :param max_attempts: maximum number of download attempts.
    :param kwargs: arguments passed to the loader.
    :return: downloaded GDAL dataset.
    """
    if max_attempts < 1:
        raise ValueError('max_attempts must be greater than zero')
    wait_seconds = 10
    for attempt in range(max_attempts):
        try:
            return loader(**kwargs)
        except RuntimeError:
            if attempt == max_attempts - 1:
                if verbose:
                    _print(f'Failed {label} after {max_attempts} attempts.')
                raise
            if verbose:
                _print(f'Failed {label}; retrying in {wait_seconds} seconds.')
            time.sleep(wait_seconds)
            wait_seconds *= 2
    raise RuntimeError(f'Unable to download {label}')



def _tile_origins(
    bounds: tuple[float, float, float, float],
    size: float,
) -> list[tuple[float, float]]:
    """Return aligned top-left tile origins covering horizontal bounds."""
    min_x, min_y, max_x, max_y = bounds
    first_x = np.floor(min_x / size) * size
    first_y = np.ceil(max_y / size) * size
    x_values = np.arange(first_x, max_x, size)
    y_values = np.arange(first_y, min_y, -size)
    return [(float(x), float(y)) for x in x_values for y in y_values]



def _tile_corners(origin: tuple[float, float], size: float) -> tuple[list[float], ...]:
    """Return the four horizontal corners of an aligned tile."""
    x, y = origin
    return ([x, y], [x + size, y], [x, y - size], [x + size, y - size])



def _download_cityjson_tile(
    tiles_dir: str,
    name: str,
    epsg: int,
    origin: tuple[float, float],
    size: float,
    spacial_resolution: float,
    verbose: bool,
) -> str:
    """Download and save imagery, DTM, and DSM for one tile."""
    datasets = (
        (const.ORTHOPHOTO_SUBDIR, _download_with_retry(
            sat.load_imagery, 'imagery', verbose, epsg=epsg, origin=origin, dimension=size,
            pixel_size=spacial_resolution,
        )),
        (const.DTM_SUBDIR, _download_with_retry(
            sat.load_dtm, 'DTM', verbose, epsg=epsg, origin=origin, dimension=size,
            pixel_size=spacial_resolution,
        )),
        (const.DSM_SUBDIR, _download_with_retry(
            sat.load_dsm, 'DSM', verbose, epsg=epsg, origin=origin, dimension=size,
            pixel_size=spacial_resolution,
        )),
    )
    image_path = ''
    for subdir, dataset in datasets:
        output_path = os.path.join(tiles_dir, subdir, f'{name}.tif')
        geo.save_geotiff(dataset, output_path)
        if subdir == const.ORTHOPHOTO_SUBDIR:
            image_path = output_path
    return image_path



def load_tile_in_cityjson(
    tiles_dir: str,
    cityjson_path: str,
    size: float = 40,
    spacial_resolution: float = 0.3,
    bounding_box: bool = False,
    skip_existing: bool = True,
    verbose: bool = False,
) -> list[str]:
    """
    Download raster tiles covering a CityJSON model.

    :param tiles_dir: directory where downloaded tile subdirectories are stored.
    :param cityjson_path: path to the CityJSON model used for tile selection.
    :param size: tile width and height in coordinate units.
    :param spacial_resolution: output pixel size in coordinate units per pixel.
    :param bounding_box: whether CityJSON containment should use its bounding box.
    :param skip_existing: skip tiles whose imagery file already exists.
    :param verbose: print tiles as they are processed, skipped, or retried.
    :return: paths to the downloaded or existing imagery tiles.
    """
    if size <= 0 or spacial_resolution <= 0:
        raise ValueError('size and spacial_resolution must be greater than zero')
    city = cjio.read_cityjson(cityjson_path)
    epsg = city.epsg()
    if epsg is None:
        raise ValueError('CityJSON must define an EPSG reference system')
    os.makedirs(tiles_dir, exist_ok=True)
    city_bounds = city.get_bounding_box()
    if city_bounds is None:
        raise ValueError('CityJSON does not contain any vertices')
    min_y, max_y, min_x, max_x = city_bounds
    bounds = min_x, min_y, max_x, max_y
    paths: list[str] = []
    for origin in _tile_origins(bounds, size):
        if not any(city.is_coordinate_inside(corner, bounding_box)
                   for corner in _tile_corners(origin, size)):
            continue
        x, y = origin
        name = f'{int(size)}_{int(x)}_{int(y)} ({date.today().isoformat()})'
        image_path = os.path.join(tiles_dir, const.ORTHOPHOTO_SUBDIR, f'{name}.tif')
        paths.append(image_path)
        if skip_existing and os.path.isfile(image_path):
            if verbose:
                _print(f'Skipping tile: {name}')
            continue
        if verbose:
            _print(f'Processing tile: {name}')
        try:
            _download_cityjson_tile(
                tiles_dir, name, epsg, origin, size, spacial_resolution, verbose,
            )
        except RuntimeError as error:
            if verbose:
                _print(f'Skipping failed tile: {name} ({error})')
    return paths


"""
Commands for processing geotiff files
"""


import os
from glob import glob
from typing import Callable

import imageio.v3 as iio
import numpy as np

import pydsm.geo as geo
import pydsm.nda as nda
from pydsm import const
from pydsm.geo_cmd import compute_ndsm


from pydsm.cmd import _print


def _read_tile_arrays(
    orthophoto_path: str,
    ndsm_path: str,
    clip_height: float | None,
) -> np.ndarray:
    """Read and combine orthophoto and nDSM arrays as four channels."""
    orthophoto = geo.to_ndarray(geo.open_geotiff(orthophoto_path))
    ndsm = geo.to_ndarray(geo.open_geotiff(ndsm_path))
    orthophoto = nda.to_uint8(nda.normalize(orthophoto))
    if clip_height is None:
        ndsm_array = ndsm.astype(float)
    else:
        ndsm_array = nda.to_uint8(
            nda.clip_rescale(ndsm, clip_height),
            norm=False,
        )
    if ndsm_array.ndim == 2:
        ndsm_array = ndsm_array[..., np.newaxis]
    return np.concatenate((orthophoto, ndsm_array), axis=-1)


def export_png_tile(
    orthophoto_path: str,
    ndsm_path: str,
    save_path: str,
    clip_height: float = const.CLIP_HEIGHT,
    verbose: bool = False,
) -> None:
    """
    Export an orthophoto and nDSM tile as an RGBA PNG.

    :param orthophoto_path: path to the orthophoto GeoTIFF.
    :param ndsm_path: path to the nDSM GeoTIFF.
    :param save_path: path to the output PNG.
    :param clip_height: nDSM value represented by 255.
    :param verbose: whether to print the exported path.
    :return: None.
    """
    array = _read_tile_arrays(orthophoto_path, ndsm_path, clip_height)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    iio.imwrite(save_path, array)
    if verbose:
        _print(f'Exported PNG tile to \'{save_path}\'')


def export_npz_tile(
    orthophoto_path: str,
    ndsm_path: str,
    save_path: str,
    clip_height: float | None = const.CLIP_HEIGHT,
    verbose: bool = False,
) -> None:
    """
    Export an orthophoto and nDSM tile as a four-channel NumPy archive.

    :param orthophoto_path: path to the orthophoto GeoTIFF.
    :param ndsm_path: path to the nDSM GeoTIFF.
    :param save_path: path to the output NPZ.
    :param clip_height: nDSM value represented by 255, or None for floats.
    :param verbose: whether to print the exported path.
    :return: None.
    """
    array = _read_tile_arrays(orthophoto_path, ndsm_path, clip_height)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    nda.write_numpy(save_path, data=array)
    if verbose:
        _print(f'Exported NPZ tile to \'{save_path}\'')


def _export_tiles(
    tiles_dir: str,
    output_subdir: str,
    exporter: Callable[[str, str, str, float | None, bool], None],
    clip_height: float | None,
    verbose: bool,
) -> list[str]:
    """Export all matching orthophoto and nDSM tiles."""
    orthophoto_dir = os.path.join(tiles_dir, const.ORTHOPHOTO_SUBDIR)
    ndsm_dir = os.path.join(tiles_dir, const.NDSM_SUBDIR)
    output_dir = os.path.join(tiles_dir, output_subdir)
    paths: list[str] = []
    for orthophoto_path in sorted(glob(os.path.join(orthophoto_dir, '*.tif'))):
        name = os.path.splitext(os.path.basename(orthophoto_path))[0]
        ndsm_path = os.path.join(ndsm_dir, f'{name}.tif')
        if not os.path.isfile(ndsm_path):
            continue
        extension = '.png' if output_subdir == const.RGBA_PNG_TILES_SUBDIR else '.npz'
        save_path = os.path.join(output_dir, f'{name}{extension}')
        exporter(orthophoto_path, ndsm_path, save_path, clip_height, verbose)
        paths.append(save_path)
    return paths


def export_png_tiles(
    tiles_dir: str,
    clip_height: float = const.CLIP_HEIGHT,
    verbose: bool = False,
) -> list[str]:
    """
    Export all orthophoto and nDSM tiles as RGBA PNG files.

    :param tiles_dir: directory containing orthophoto and nDSM subdirectories.
    :param clip_height: nDSM value represented by 255.
    :param verbose: whether to print exported paths.
    :return: paths to the exported PNG files.
    """
    return _export_tiles(
        tiles_dir, const.RGBA_PNG_TILES_SUBDIR, export_png_tile, clip_height, verbose,
    )


def export_npz_tiles(
    tiles_dir: str,
    clip_height: float | None = const.CLIP_HEIGHT,
    verbose: bool = False,
) -> list[str]:
    """
    Export all orthophoto and nDSM tiles as four-channel NPZ files.

    :param tiles_dir: directory containing orthophoto and nDSM subdirectories.
    :param clip_height: nDSM value represented by 255, or None for floats.
    :param verbose: whether to print exported paths.
    :return: paths to the exported NPZ files.
    """
    return _export_tiles(
        tiles_dir, const.RGBA_NPZ_TILES_SUBDIR, export_npz_tile, clip_height, verbose,
    )


def compute_ndsm_dir(
    tiles_dir: str,
    correct_dtm: bool = False,
    capture_height: float = 500.0,
    cmap_dir: str | None = None,
    verbose: bool = False,
) -> None:
    """
    Compute nDSM files for matching DSM and DTM files in a tiles directory.

    :param tiles_dir: directory containing DSM and DTM subdirectories.
    :param correct_dtm: whether to correct extreme DTM values.
    :param capture_height: maximum vegetation height to retain.
    :param cmap_dir: optional subdirectory for RGB nDSM renderings.
    :param verbose: whether to print processing messages.
    """
    dsm_dir = os.path.join(tiles_dir, const.DSM_SUBDIR)
    dtm_dir = os.path.join(tiles_dir, const.DTM_SUBDIR)
    ndsm_dir = os.path.join(tiles_dir, const.NDSM_SUBDIR)

    os.makedirs(ndsm_dir, exist_ok=True)

    if cmap_dir is not None:
        os.makedirs(os.path.join(tiles_dir, cmap_dir), exist_ok=True)
    for dsm_path in sorted(glob(os.path.join(dsm_dir, '*.tif'))):
        filename = os.path.basename(dsm_path)
        dtm_path = os.path.join(dtm_dir, filename)
        if not os.path.isfile(dtm_path):
            if verbose:
                _print(f'Skipping \'{dsm_path}\': matching DTM not found')
            continue
        ndsm_path = os.path.join(ndsm_dir, filename)
        output_path = compute_ndsm(
            dsm_path,
            dtm_path,
            ndsm_path,
            correct_dtm,
            capture_height,
            verbose=verbose,
        )
        if cmap_dir is not None:
            ndsm = geo.open_geotiff(output_path)
            cmap_path = os.path.join(tiles_dir, cmap_dir, f'{os.path.splitext(filename)[0]}.png')
            iio.imwrite(cmap_path, geo.to_cmap(ndsm, clip_height=capture_height))



def extract_tiles(orthophoto_path: str, tiles_dir: str, date: str, tile_size: int = None, scale: float = None, verbose: bool = True, *, ndsm_path: str = None, dsm_path: str = None) -> None:
    """
    Extract valid tiles from an orthophoto and NDSM, and save them to a directory.
    Tiles name : `{tile_size}_{x}_{y} ({date}) ({distance from border}m).tif`
        Bigger distances from the border should indicate better quality tiles.

    :param orthophoto_path: path to the orthophoto geotiff file
    :param ndsm_path: path to the NDSM geotiff file
    :param tiles_dir: directory to save the extracted tiles (must contain a 'orthophoto' and 'ndsm' subdirectory)
    :param date: date of the tiles (used in the naming convention of the saved tiles)
    :param tile_size: size of the tiles in meters (default is 50m)
    :param scale: scale in meters per pixel to resize the tiles (default is 0.02)
    :param verbose: whether to display the tile grid (default is True)
    :return: None (saves the tiles to the directory)    
    """
    tile_size = tile_size or 50  # default tile size in meters
    scale = scale or 0.02  # default scale in meters per pixel

    _print(f'* Extracting tiles from \'{orthophoto_path}\'')
    orthophoto = geo.open_geotiff(orthophoto_path)
    tiles_coordinates, tile_distances = geo.get_tiles_coordinates(orthophoto, tile_size, scale)

    if verbose:
        geo.display_tile_grid(orthophoto, tile_size, tiles_coordinates=tiles_coordinates)

    _print('* Cropping orthophoto into tiles')
    array = geo.to_ndarray(orthophoto)
    ortho_dir = os.path.join(tiles_dir, 'orthophoto')
    for i in range(len(tiles_coordinates)):
        tile = geo.crop_into_tile(orthophoto, tiles_coordinates[i], tile_size, scale, array)
        geo.save_tile(tile, tile_distances[i], ortho_dir, date)
    _print(f'  Tiles saved in \'{ortho_dir}\'')

    if ndsm_path:
        _print('* Cropping NDSM into tiles')
        ndsm = geo.open_geotiff(ndsm_path)
        array = geo.to_ndarray(ndsm)
        ndsm_dir = os.path.join(tiles_dir, 'ndsm')
        for i in range(len(tiles_coordinates)):
            tile = geo.crop_into_tile(ndsm, tiles_coordinates[i], tile_size, scale, array)
            geo.save_tile(tile, tile_distances[i], ndsm_dir, date)
        _print(f'  Tiles saved in \'{ndsm_dir}\'')

    if dsm_path:
        _print('* Cropping DSM into tiles')
        ndsm = geo.open_geotiff(ndsm_path)
        array = geo.to_ndarray(ndsm)
        dsm_dir = os.path.join(tiles_dir, 'dsm')
        for i in range(len(tiles_coordinates)):
            tile = geo.crop_into_tile(ndsm, tiles_coordinates[i], tile_size, scale, array)
            geo.save_tile(tile, tile_distances[i], dsm_dir, date)
        _print(f'  Tiles saved in \'{dsm_dir}\'')

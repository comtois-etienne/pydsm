"""
Commands for processing geotiff files
"""


import os
from glob import glob

import imageio.v3 as iio

import pydsm.geo as geo
from pydsm import const
from pydsm.geo_cmd import compute_ndsm


from pydsm.cmd import _print


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

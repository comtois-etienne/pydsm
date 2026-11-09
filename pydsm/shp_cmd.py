"""
Commands for processing geotiff files
"""



import pandas as pd

import pydsm.geo as geo
import pydsm.shp as shp
import pydsm.utils as utils


from pydsm.cmd import _print


def csv_to_shapefile(csv_path: str, shapefile_path: str = None, epsg: int = None):
    """
    Creates a shapefile from a CSV file

    :param csv_path: str, path to the CSV file (mandatory)
        contains a comment line with the EPSG code (optional)
        For epsg=4326, the longitude is 'x' and the latitude is 'y'
        csv example:
            ```
            #epsg=4326
            x,y
            0,0
            ```
    :param shapefile_path: str, path to save the shapefile (mandatory)
    :param epsg: int, EPSG code for the coordinate system (optional)
    """
    shp_path = shapefile_path or f'{utils.remove_extension(csv_path)}.shp'
    shp.save_from_csv(csv_path, shp_path, epsg=epsg)
    _print(f'* Saved to \'{shapefile_path}\'')



def crop_geotiff(geotiff_path: str, shapefile_path: str, save_path: str = None, dilate: float = 15.0):
    """
    Crop a geotiff file with a shapefile

    :param geotiff_path: str, path to the geotiff file (mandatory)
    :param shapefile_path: str, path to the shapefile (mandatory)
    :param save_path: str, path to save the cropped geotiff (optional)
    """
    path = utils.get_folder_path(geotiff_path)
    uuid = utils.remove_extension(utils.get_filename(shapefile_path))
    name = utils.remove_extension(utils.get_filename(geotiff_path))

    if save_path:
        if utils.get_extension(save_path) == '':
            save_path = utils.append_file_to_path(save_path, f'{uuid}_{name}.tif')
    else: 
        save_path = utils.append_file_to_path(path, f'{uuid}_{name}.tif')

    dilate = dilate or 15.0
    gdal = geo.open_geotiff(geotiff_path)
    cropped = geo.crop_from_shapefile(gdal, shapefile_path, dilate_size=dilate)
    geo.save_geotiff(cropped, save_path)
    _print(f'* Saved to \'{save_path}\'')



def extract_by_zone(geotiff_paths: list,
                    geotiff_base_path: str = None,
                    save_directory: str = None,
                    no_crop: bool = False,
                    dilate: float = 15.0,
                    safe_zone: float = 20.0,
                    sample_size: int = 3,
                    translate_x: float = 0.0,
                    translate_y: float = 0.0,
                    translate_file: str = None):
    """
    Extracts zones surrounded by streets from a geotiff file
    Crops the geotiff file to the zones if --no-crop is not set and --crop-geotiff is not set

    :param geotiff_paths: str, path to the geotiff file (mandatory)
    :param geotiff_base_path: str, path to the folder that contains all the geotiff files (optional)
    :param save_directory: str, path to save the zones (shapefile, geotiff) (optional)
    :param no_crop: bool, do not crop the geotiff to the zones (optional)
    :param dilate: float, dilation factor around the zones (optional) (default: 15.0m)
    :param safe_zone: float, the zone is only kept if there is at lest X meters around (optional) (default: 20.0m)
    :param sample_size: int, sample size (number of seeds) to find the zones (optional) (default: 3 (3x3))
    :param translate_x: float, X translation (East to West) (optional)
    :param translate_y: float, Y translation (South to North) (optional)
    :param translate_file: str, path that contains the translation values (optional)
    :return: None (saves the zones to disk as geotiffs)
    """
    if geotiff_base_path:
        geotiff_paths = [utils.append_file_to_path(geotiff_base_path, geotiff_path) for geotiff_path in geotiff_paths]

    save_folder = save_directory or utils.get_folder_path(geotiff_paths[0])
    sample_size = sample_size or 3
    safe_zone = safe_zone or 20.0
    dilate = dilate or 15.0
    x = translate_x or 0.0
    y = translate_y or 0.0

    if translate_file:
        df = pd.read_csv(translate_file)
        x = df['x'][0]
        y = df['y'][0]

    _print(f'* Extracting zones from \'{geotiff_paths[0]}\'')
    uuids = geo.extract_zones(
        geotiff_paths[0], 
        save_folder, 
        sample_size=sample_size, 
        safe_zone=safe_zone
    )
    _print(f'* Saved to {save_folder}')

    geotiff_paths = [] if no_crop else geotiff_paths
    for geotiff_path in geotiff_paths:
        _print(f'* Cropping \'{geotiff_path}\'')
        geotiff_file = geo.open_geotiff(geotiff_path)

        if x or y:
            _print(f'  Translating by {x}m X and {y}m Y')
            geotiff_file = geo.translation(geotiff_file, (x, y))

        name = utils.get_filename(geotiff_path)
        name = utils.remove_extension(name)

        for uuid in uuids:
            save_path = utils.append_file_to_path(save_folder, uuid)
            file_path = f'{save_path}_{name}.tif'
            croped_ortho = geo.crop_from_shapefile(geotiff_file, f'{save_path}.shp', dilate_size=dilate)
            geo.save_geotiff(croped_ortho, file_path)
            _print(f'  Saved to \'{file_path}\'')



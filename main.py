#!/opt/anaconda3/envs/geo/bin/python

import argparse
import time
import os
import sys


# COMMANDS

def _ndsm(args):
    """
    Computes the nDSM from a DSM and DTM

    :param args.dsm_path: str, path to the DSM (mandatory)
    :param args.dtm_path: str, path to the DTM (mandatory)
    :param args.ndsm_path: str, path to save the nDSM (optional)
    :param args.correct_dtm: bool, removes extreme DTM values (optional)
    :param args.capture_height: float, height of the drone during the capture (optional) (default: 60.0)
    :param args.resize: str, path to the orthophoto to resize the DSM and DTM (optional)
    :return: None (saves the nDSM to disk)
    """
    from pydsm.cmd import compute_ndsm
    compute_ndsm(
        args.dsm_path, 
        args.dtm_path, 
        args.ndsm_path, 
        args.correct_dtm, 
        args.capture_height, 
        args.resize
    )


def _epsg(args):
    """
    Displays the EPSG code of a file

    :param args.path: str, path to the file (mandatory)
    :return: None (prints the EPSG code to the console)
    """
    from pydsm.cmd import display_epsg
    display_epsg(args.path)


def _reproject(args):
    """
    Reprojects a geotiff file to a new EPSG code

    :param args.path: str, path to the geotiff file (mandatory)
    :param args.save_path: str, path to save the reprojected file (optional)
    :param args.epsg: int, new EPSG code (optional) (default: 4326)
    :return: None (saves the reprojected file to disk)
    """
    from pydsm.cmd import reproject_geotiff
    reproject_geotiff(args.path, args.save_path, args.epsg)


def _xyz(args):
    """
    Calculates the XYZ coordinates from a geotiff file (dtm, dsm, ndsm, etc.)
    Not recommended for large files. Use the geotiff directly instead.

    :param args.path: str, path to the geotiff file (mandatory)
    :param args.save_path: str, path to save the XYZ file (optional)
    :return: None (saves the XYZ file to disk as a compressed numpy file (.npz))
    """
    from pydsm.cmd import to_xyz
    to_xyz(args.path, args.save_path)


def _cmap(args):
    """
    Converts a geotiff (dsm, dtm, ndsm) file to a colormap png

    :param args.path: str, path to the geotiff file (mandatory)
    :param args.cmap: str, colormap name (optional) (default: viridis)
    :param args.save_path: str, path to save the colormap (optional)
    :return: None (saves the colormap to disk)
    """
    from pydsm.cmd import to_cmap
    to_cmap(args.path, args.cmap, args.save_path)


def _values(args):
    """
    Overlays the pixel values over each pixel of a cropped portion of an image

    :param args.path: str, path to the image file (mandatory)
    :param args.y: int, Y coordinate of the top left corner (mandatory)
    :param args.x: int, X coordinate of the top left corner (mandatory)
    :param args.h: int, height of the cropped image (mandatory)
    :param args.w: int, width of the cropped image (mandatory)
    :param args.cmap: str, colormap name (optional) (default: viridis)
    :param args.save_path: str, path of the croped region (optional)
    :param args.region: bool, overlay the region over the original image (optional)
    :param args.downscale: int, downscale factor for the image for performance reasons (optional)
    :return: None (saves the overlayed image to disk)
    """
    from pydsm.cmd import overlay_values
    overlay_values(
        args.path, 
        args.y, 
        args.x, 
        args.h, 
        args.w, 
        args.cmap, 
        args.save_path, 
        args.region, 
        args.downscale
    )


def _shapefile(args):
    """
    Creates a shapefile from a CSV file

    :param args.csv_path: str, path to the CSV file (mandatory)
        contains a comment line with the EPSG code (optional)
        For epsg=4326, the longitude is 'x' and the latitude is 'y'
        csv example:
            ```
            #epsg=4326
            x,y
            0,0
            ```
    :param args.shapefile_path: str, path to save the shapefile (mandatory)
    :param args.epsg: int, EPSG code for the coordinate system (optional)
    """
    from pydsm.cmd import csv_to_shapefile
    csv_to_shapefile(args.csv_path, args.shapefile_path, args.epsg)


def _crop(args):
    """
    Crop a geotiff file with a shapefile

    :param args.geotiff_path: str, path to the geotiff file (mandatory)
    :param args.shapefile_path: str, path to the shapefile (mandatory)
    :param args.save_path: str, path to save the cropped geotiff (optional)
    :param args.dilate: float, dilation factor around the shapefile (optional) (default: 15.0m)
    """
    from pydsm.cmd import crop_geotiff
    crop_geotiff(args.geotiff_path, args.shapefile_path, args.save_path, args.dilate)


def _zones(args):
    """
    Extracts zones surrounded by streets from a geotiff file
    Crops the geotiff file to the zones if --no-crop is not set and --crop-geotiff is not set

    :param args.geotiff_paths: str, path to the geotiff file (mandatory)
    :param args.geotiff_base_path: str, path to the folder that contains all the geotiff files (optional)
    :param args.save_directory: str, path to save the zones (shapefile, geotiff) (optional)
    :param args.no_crop: bool, do not crop the geotiff to the zones (optional)
    :param args.dilate: float, dilation factor around the zones (optional) (default: 15.0m)
    :param args.safe_zone: float, the zone is only kept if there is at lest X meters around (optional) (default: 20.0m)
    :param args.sample_size: int, sample size (number of seeds) to find the zones (optional) (default: 3 (3x3))
    :param args.translate_x: float, X translation (East to West) (optional)
    :param args.translate_y: float, Y translation (South to North) (optional)
    :param args.translate_file: str, path that contains the translation values (optional)
    :return: None (saves the zones to disk as geotiffs)
    """
    from pydsm.cmd import extract_by_zone
    extract_by_zone(
        args.geotiff_paths, 
        args.geotiff_base_path, 
        args.save_directory, 
        args.no_crop, 
        args.dilate, 
        args.safe_zone, 
        args.sample_size, 
        args.translate_x, 
        args.translate_y, 
        args.translate_file
    )


def _info(args):
    """
    Displays information about a geotiff file

    :param args.path: str, path to the geotiff file (mandatory)
    :return: None (displays the information to the console)
    """
    from pydsm.cmd import display_geotiff_info
    display_geotiff_info(args.path)


def _resize(args):
    """
    Resize a geotiff file  
    `Warning` the resized geotiff will break the georeferencing

    :param args.geotiff_path: str, path to the geotiff file (mandatory)
    :param args.geotiff_like_path: str, path to the geotiff file to get the shape from (mandatory)
    :param args.save_path: str, path to save the resized (optional)
    :return: None (saves the resized geotiff to disk)
    """
    from pydsm.cmd import resize_geotiff_like
    resize_geotiff_like(args.geotiff_path, args.geotiff_like_path, args.save_path)


def _rescale(args):
    """
    Rescale a geotiff file from its original scale to a new scale

    :param args.geotiff_path: str, path to the geotiff file (mandatory)
    :param args.scale: float, scale factor in meters/px (mandatory)
    :param args.save_path: str, path to save the rescaled (optional)
    :return: None (saves the rescaled geotiff to disk)
    """
    from pydsm.cmd import rescale_geotiff
    rescale_geotiff(args.geotiff_path, args.scale, args.save_path)


def _translation(args):
    """
    2D translation of a geotiff file

    :param args.geotiff_path: str, path to the geotiff file (mandatory)
    :param args.x: float, X translation (mandatory)
    :param args.y: float, Y translation (mandatory)
    :param args.save_path: str, path to save the translated geotiff (optional)
    :return: None (saves the translated geotiff to disk)
    """
    from pydsm.cmd import translate_geotiff
    translate_geotiff(args.geotiff_path, args.x, args.y, args.save_path)


def registration(args):
    """
    Register a geotiff file onto the osm map using a translation  
    The geotiff file is translated to the selected point on the map  

    :param args.geotiff_path: str, path to the geotiff file (mandatory)
    :param args.save_path: str, path to save the registered geotiff (optional)
    :param translate: bool, translate the geotiff to the selected point (False will simply save the translation values) (optional)
    :param args.registration_path: str, path to save the translation values (optional)
    :param args.layer: str, layer of the map (osm, streets, satellite) (optional) (default: osm)
    :return: None (saves the registered geotiff to disk and the translation values to a csv file)
    """
    from pydsm.cmd import geotiff_registration
    _unsilence()
    geotiff_registration(
        args.geotiff_path, 
        args.save_path, 
        args.translate, 
        args.registration_path, 
        args.layer
    )


def RBH(args):
    """
    Generates geometry objects for trees from the masks and the ndsm.  
    The triangulated rhombicuboctahedron (RBH) geometry is used to represent the trees. 
    The dtm is used to get the ground level of the trees.
    The orthophoto is used to get the rejection region of the trees. 

    :param args.masks_path: str, path to the instance masks (mandatory)
    :param args.ndsm_path: str, path to the nDSM (mandatory)
    :param args.dtm_path: str, path to the DTM (mandatory)
    :param args.ortho_path: str, path to the orthophoto (mandatory)
    :param args.save_dir: str, path to directory to save the wavefront and metadata (optional) (default: current directory)
    :param args.elevation_offset: float, elevation offset for the trees instead of relying on the DTM (optional) (default: False)
    :param args.verbose: bool, verbose mode to display the whole process (optional) (default: False)
    :return: None (saves the wavefront and metadata to disk)
    """
    from pydsm.cmd import generate_rbh_wavefront
    generate_rbh_wavefront(
        args.masks,
        args.ndsm,
        args.dtm,
        args.ortho,
        args.save_dir,
        args.offset,
        args.verbose
    )


# GENERAL COMMANDS

def _silence():
    sys.stdout = open(os.devnull, 'w')


def _unsilence():
    sys.stdout = sys.__stdout__


def silent_mode(args):
    if args.silent:
        _silence()


def time_mode(args, t0):
    _unsilence()
    t1 = time.time()
    if args.time:
        print(f'* Elapsed time: {t1 - t0:.2f} seconds')


# ARGUMENT PARSER SETUP

COMMANDS = {
    'ndsm': _ndsm,
    'epsg': _epsg,
    'reproject': _reproject,
    'xyz': _xyz,
    'cmap': _cmap,
    'values': _values,
    'shapefile': _shapefile,
    'crop': _crop,
    'zones': _zones,
    'info': _info,
    'resize': _resize,
    'rescale': _rescale,
    'translation': _translation,
    'registration': registration,
    'rbh': RBH,
}


def parser_setup():
    parser = argparse.ArgumentParser(description='PyDSM : DSM, DTM, nDSM & Ortophoto tools')
    parser.add_argument("--silent", action="store_true", help="Silent mode")
    parser.add_argument("--time", action="store_true", help="Displays time elapsed")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ndsm command
    parser_ndsm = subparsers.add_parser("ndsm", help="Generate a nDSM from a DSM and DTM")
    parser_ndsm.add_argument("dsm_path", type=str, help="Path to the DSM (mandatory)")
    parser_ndsm.add_argument("dtm_path", type=str, help="Path to the DTM (mandatory)")
    parser_ndsm.add_argument("--ndsm_path", type=str, help="Path to the output nDSM")
    parser_ndsm.add_argument("--correct-dtm", action="store_true", help="Correct extreme DTM values")
    parser_ndsm.add_argument("--capture-height", type=float, help="Height of the drone during the capture (default: 60.0)")
    parser_ndsm.add_argument("--resize", type=str, help="Resize the DSM and DTM to the orthophoto shape")

    # epsg command
    parser_epsg = subparsers.add_parser("epsg", help="Get the EPSG code of a file (GeoTIFF or Shapefile)")
    parser_epsg.add_argument("path", type=str, help="Path to the file")

    # reproject command
    parser_reproject = subparsers.add_parser("reproject", help="Reproject a file to a new EPSG code")
    parser_reproject.add_argument("epsg", type=int, help="New EPSG code")
    parser_reproject.add_argument("path", type=str, help="Path to the geotiff file")
    parser_reproject.add_argument("--save-path", type=str, help="Path to save the reprojected file")

    # xyz command
    parser_xyz = subparsers.add_parser("xyz", help="Generate a XYZ file from a geotiff as a compressed numpy (.npz)")
    parser_xyz.add_argument("path", type=str, help="Path to the geotiff file")
    parser_xyz.add_argument("--save-path", type=str, help="Path to save the XYZ file")

    # cmap command
    parser_cmap = subparsers.add_parser("cmap", help="Generate a colormap png from a geotiff")
    parser_cmap.add_argument("path", type=str, help="Path to the geotiff file")
    parser_cmap.add_argument("--cmap", type=str, help="Colormap name (default: viridis)")
    parser_cmap.add_argument("--save-path", type=str, help="Path to save the colormap")

    # values command
    parser_values = subparsers.add_parser("values", help="Displays the values of a cropped image file (GeoTIFF or PNG)")
    parser_values.add_argument("path", type=str, help="Path to the image file")
    parser_values.add_argument("y", type=int, help="Y coordinate of the top left corner")
    parser_values.add_argument("x", type=int, help="X coordinate of the top left corner")
    parser_values.add_argument("h", type=int, help="height of the cropped image")
    parser_values.add_argument("w", type=int, help="width of the cropped image")
    parser_values.add_argument("--cmap", type=str, help="Colormap name (default: viridis)")
    parser_values.add_argument("--save-path", type=str, help="Path to save the values")
    parser_values.add_argument("--region", action="store_true", help="Overlay the region on the original image")
    parser_values.add_argument("--downscale", type=int, help="Downscale factor for the image")

    # create shapefile from csv
    parser_shp_create = subparsers.add_parser("shapefile", help="Create a shapefile from a CSV file")
    parser_shp_create.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser_shp_create.add_argument("--shapefile_path", type=str, help="Path to save the shapefile")
    parser_shp_create.add_argument("--epsg", type=int, help="EPSG code for the coordinate system")

    # crop geotiff with shapefile
    crop_parser = subparsers.add_parser("crop", help="Crop a geotiff file with a shapefile")
    crop_parser.add_argument("geotiff_path", type=str, help="Path to the geotiff file")
    crop_parser.add_argument("shapefile_path", type=str, help="Path to the shapefile")
    crop_parser.add_argument("--dilate", type=float, help="Dilation factor around the shapefile (default: 15.0m)")
    crop_parser.add_argument("--save-path", type=str, help="Path to save the cropped geotiff")

    # zones command
    zones_parser = subparsers.add_parser("zones", help="Crops a geotiff into zones surrounded by streets")
    zones_parser.add_argument("geotiff_paths", nargs='+', help="Path to the geotiff files. The first one is used to find the zones.")
    zones_parser.add_argument("--geotiff-base-path", type=str, help="Path to the folder that contains all the geotiff files")
    zones_parser.add_argument("--save-directory", type=str, help="Path to save the zones (shapefile, geotiff)")
    zones_parser.add_argument("--no-crop", action="store_true", help="Do not crop the geotiff to the zones") #todo default to false
    zones_parser.add_argument("--dilate", type=float, help="Dilation factor around the zones (default: 15.0m)")
    zones_parser.add_argument("--safe-zone", type=float, help="The zone is only kept if there is at least X meters around it (default: 20.0m)")
    zones_parser.add_argument("--sample-size", type=int, help="Sample size (number of seeds) to find the zones (default: 3 (3x3))")
    zones_parser.add_argument("--translate-x", type=float, help="X translation (East to West)")
    zones_parser.add_argument("--translate-y", type=float, help="Y translation (South to North)")
    zones_parser.add_argument("--translate-file", type=str, help="Path that contains the translation values (Overrides translate-x and translate-y) (default: translate.csv in geotiff[0] dir)")

    # file info command
    file_info_parser = subparsers.add_parser("info", help="Display information about a geotiff")
    file_info_parser.add_argument("path", type=str, help="Path to the geotiff")

    # resize command
    resize_parser = subparsers.add_parser("resize", help="Resize a geotiff file")
    resize_parser.add_argument("geotiff_path", type=str, help="Path to the geotiff file to resize")
    resize_parser.add_argument("geotiff_like_path", type=str, help="Path to the geotiff file to get the shape from")
    resize_parser.add_argument("--save-path", type=str, help="Path to save the resized geotiff")

    # rescale command
    rescale_parser = subparsers.add_parser("rescale", help="Rescale a geotiff file")
    rescale_parser.add_argument("scale", type=float, help="Scale factor in meters/px")
    rescale_parser.add_argument("geotiff_path", type=str, help="Path to the geotiff file to rescale")
    rescale_parser.add_argument("--save-path", type=str, help="Path to save the rescaled geotiff")

    # translation command
    translation_parser = subparsers.add_parser("translation", help="Translate a geotiff file")
    translation_parser.add_argument("geotiff_path", type=str, help="Path to the geotiff file to translate")
    translation_parser.add_argument("x", type=float, help="X translation (East to West)")
    translation_parser.add_argument("y", type=float, help="Y translation (South to North)")
    translation_parser.add_argument("--save-path", type=str, help="Path to save the translated geotiff")

    # registration command
    registration_parser = subparsers.add_parser("registration", help="Register a geotiff file onto the osm map")
    registration_parser.add_argument("geotiff_path", type=str, help="Path to the geotiff file to register")
    registration_parser.add_argument("--save-path", type=str, help="Path to save the registered geotiff")
    registration_parser.add_argument("--translate", action="store_true", help="Translate the geotiff to the selected point")
    registration_parser.add_argument("--registration-path", type=str, help="Path to save the translation values")
    registration_parser.add_argument("--layer", type=str, help="Layer of the map (osm, streets, satellite)")

    # rbh command
    rbh_parser = subparsers.add_parser("rbh", help="Create a wavefront file containing RBH shaped trees from geotiffs and instance masks")
    rbh_parser.add_argument("--masks", "--masks-path", type=str, help="Path to the masks instances", required=True)
    rbh_parser.add_argument("--ndsm", "--ndsm-path", type=str, help="Path to the nDSM file", required=True)
    rbh_parser.add_argument("--dtm", "--dtm-path", type=str, help="Path to the DTM file", required=True)
    rbh_parser.add_argument("--ortho", "--ortho-path", type=str, help="Path to the orthophoto file containing the alpha channel", required=True)
    rbh_parser.add_argument("--offset", "--elevation-offset", type=float, help="Elevation offset for the trees - The DTM values will be used if no offset is given", default=False)
    rbh_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode - Show the progression of each tree", default=False)
    rbh_parser.add_argument("--save-dir", type=str, help="Path to save the wavefront file", default='./')

    return parser


def main():
    t0 = time.time()
    parser = parser_setup()
    args = parser.parse_args()
    silent_mode(args)
    COMMANDS[args.command](args)
    time_mode(args, t0)


if __name__ == "__main__":
    main()


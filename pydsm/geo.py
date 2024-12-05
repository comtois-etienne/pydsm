import osgeo
from osgeo import gdal, ogr, osr
import numpy as np
from skimage import draw

from .nda import to_gdal as nda_to_gdal
from .nda import round_to_mm as nda_round_to_mm
from .nda import to_wavefront as nda_to_wavefront
from .shp import get_coords as shp_get_coords


def open(path: str) -> osgeo.gdal.Dataset:
    """
    Opens a geotiff file
    
    :param path: path to the file
    :return: gdal dataset
    """
    return gdal.Open(path)


def save(gdal_file: osgeo.gdal.Dataset, path: str) -> None:
    """
    Saves the gdal dataset to a compressed geotiff file
    
    :param gdal_file: gdal dataset
    :param path: path to the file
    """
    driver = gdal.GetDriverByName('GTiff')
    options = ["COMPRESS=LZW", "TILED=YES"]
    driver.CreateCopy(path, gdal_file, options=options)


# PROPRIETIES

def get_epsg(gdal_file: osgeo.gdal.Dataset) -> int:
    """
    :param gdal_file: gdal dataset
    :return: EPSG code of the dataset
    """
    proj = osr.SpatialReference(wkt=gdal_file.GetProjection())
    epsg: osgeo.osr.SpatialReference = proj.GetAttrValue('AUTHORITY', 1)
    return int(epsg)


def get_origin(gdal_file: osgeo.gdal.Dataset) -> tuple:
    """
    :param gdal_file: gdal dataset
    :return: origin of the dataset (x, y) in the coordinate system
    """
    return gdal_file.GetGeoTransform()[0], gdal_file.GetGeoTransform()[3]


def get_pixel_size(gdal_file: osgeo.gdal.Dataset) -> tuple:
    """
    :param gdal_file: gdal dataset
    :return: dimension of the pixel in the coordinate system
    """
    return gdal_file.GetGeoTransform()[1], gdal_file.GetGeoTransform()[5]


# COORDINATES

def get_coordinate_at_pixel(gdal_file: osgeo.gdal.Dataset, x: int, y: int) -> tuple:
    """
    :param gdal_file: gdal dataset
    :param x: x coordinate in pixel
    :param y: y coordinate in pixel
    :return: coordinate of the pixel (x, y) in the coordinate system
    """
    origin = get_origin(gdal_file)
    pixel_size = get_pixel_size(gdal_file)
    w, h = gdal_file.RasterXSize, gdal_file.RasterYSize
    x = w - x if x < 0 else x
    y = h - y if y < 0 else y
    return origin[0] + x * pixel_size[0], origin[1] + y * pixel_size[1]


def get_coordinates_at_pixels(gdal_file: osgeo.gdal.Dataset) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :return: array of the coordinates of the pixels in the coordinate system (x, y)
    """
    origin = get_origin(gdal_file)
    pixel_size = get_pixel_size(gdal_file)
    x = np.arange(gdal_file.RasterXSize)
    y = np.arange(gdal_file.RasterYSize)
    xx, yy = np.meshgrid(x, y)
    all_x = origin[0] + xx * pixel_size[0]
    all_y = origin[1] + yy * pixel_size[1]
    return np.stack((all_x, all_y), axis=-1)


def get_pixel_at_coordinate(gdal_file: osgeo.gdal.Dataset, x: float, y: float) -> tuple:
    """
    :param gdal_file: gdal dataset
    :param x: x coordinate in the coordinate system
    :param y: y coordinate in the coordinate system
    :return: pixel of the coordinate (x, y) in the dataset
    """
    origin = get_origin(gdal_file)
    pixel_size = get_pixel_size(gdal_file)
    return int((x - origin[0]) / pixel_size[0]), int((y - origin[1]) / pixel_size[1])


def get_pixels_at_coordinates(gdal_file: osgeo.gdal.Dataset, coords: np.ndarray | list) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :param coords: array of the coordinates in the coordinate system (x, y)
    :return: array of the pixels of the coordinates in the dataset
    """
    if isinstance(coords, list):
        coords = np.array(coords)
    origin = get_origin(gdal_file)
    pixel_size = get_pixel_size(gdal_file)
    pixels = np.stack((
        (coords[:, 0] - origin[0]) / pixel_size[0], 
        (coords[:, 1] - origin[1]) / pixel_size[1]
        ), axis=-1).astype(int)
    return pixels


# CONVERSIONS

def reproject(gdal_file: osgeo.gdal.Dataset, epsg: int) -> osgeo.gdal.Dataset:
    """
    Changes the coordinate system of the dataset to the given EPSG code
    
    :param gdal_file: gdal dataset
    :param epsg: desired EPSG code
    :return: gdal dataset with the new coordinate system
    """
    target = osr.SpatialReference()
    target.ImportFromEPSG(epsg)
    # transform = osr.CoordinateTransformation(gdal_file.GetSpatialRef(), target)
    return gdal.AutoCreateWarpedVRT(gdal_file, None, target.ExportToWkt(), gdal.GRA_NearestNeighbour)


def to_ndarray(gdal_file: osgeo.gdal.Dataset) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :return: array of the dataset
    """
    return gdal_file.GetRasterBand(1).ReadAsArray()


def to_xyz(gdal_file: osgeo.gdal.Dataset) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :return: array of the coordinates of the pixels in the coordinate system (x, y, z)
    """
    nda = to_ndarray(gdal_file) # z
    coordinates = get_coordinates_at_pixels(gdal_file) # x, y
    return np.dstack((coordinates, nda))


def to_gdal_like(nda: np.ndarray, gdal_like: gdal.Dataset) -> gdal.Dataset:
    """
    Converts a NumPy array to a GDAL dataset with the same metadata as the input dataset.
    
    :param nda: NumPy array (2D or 3D)
    :param gdal_like: GDAL dataset to copy metadata from
    :return: GDAL dataset
    """
    pixel_size = get_pixel_size(gdal_like)[0]
    origin = get_origin(gdal_like)
    epsg = get_epsg(gdal_like)
    return nda_to_gdal(nda, epsg, origin, pixel_size)


def to_ndsm(dsm_gdal: osgeo.gdal.Dataset, dtm_gdal: osgeo.gdal.Dataset, round_to_millimeters=True) -> np.ndarray:
    """
    :param dsm_gdal: gdal dataset of the DSM
    :param dtm_gdal: gdal dataset of the DTM
    :return: array of the NDMS
    """
    if dsm_gdal.RasterXSize != dtm_gdal.RasterXSize or dsm_gdal.RasterYSize != dtm_gdal.RasterYSize:
        raise RuntimeError("The DSM and DTM must have the same dimensions.")
    
    if get_epsg(dsm_gdal) != get_epsg(dtm_gdal):
        raise RuntimeError("The DSM and DTM must have the same coordinate system.")

    dsm = to_ndarray(dsm_gdal)
    dtm = to_ndarray(dtm_gdal)
    ndsm = dsm - dtm
    ndsm[ndsm < 0.0] = 0.0
    ndsm = nda_round_to_mm(ndsm) if round_to_millimeters else ndsm
    ndsm = to_gdal_like(ndsm, dsm_gdal)
    return ndsm


def mask_from_coords(gdal_file: osgeo.gdal.Dataset, coords: np.ndarray | list) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :param coords: array of the coordinates in the coordinate system (x, y)
    :return: mask of the coordinates in the dataset
    """
    if isinstance(coords, list):
        coords = np.array(coords)

    points = get_pixels_at_coordinates(gdal_file, coords)
    array = to_ndarray(gdal_file)
    mask = np.zeros(array.shape, dtype=bool)

    r = [point[1] for point in points]
    c = [point[0] for point in points]
    rr, cc = draw.polygon(r, c)
    mask[rr, cc] = True

    return mask


def mask_from_shapefile(gdal_file: osgeo.gdal.Dataset, shapefile_path: str) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :param shapefile_path: path to the shapefile
    :return: mask of the shapefile in the dataset
    """
    coords = shp_get_coords(shapefile_path)
    return mask_from_coords(gdal_file, coords)


def crop_from_shapefile(gdal_file: osgeo.gdal.Dataset, shapefile_path: str, mask_value=0.0) -> osgeo.gdal.Dataset:
    """
    :param gdal_file: gdal dataset
    :param shapefile_path: path to the shapefile
    :param mask_value: value to fill the mask
    :return: cropped gdal dataset from the shapefile
    """
    coords = np.array(shp_get_coords(shapefile_path))
    pixels = get_pixels_at_coordinates(gdal_file, coords)
    mask = mask_from_coords(gdal_file, coords)
    array = to_ndarray(gdal_file)
    array = array * mask
    array[~mask] = mask_value
    min_px, min_py = np.min(pixels, axis=0)
    max_px, max_py = np.max(pixels, axis=0)
    array = array[min_py:max_py, min_px:max_px]
    gdal_croped = nda_to_gdal(
        array, 
        get_epsg(gdal_file), 
        np.min(coords, axis=0), 
        get_pixel_size(gdal_file)[0]
    )
    return gdal_croped


def round_to_mm(gdal_file: osgeo.gdal.Dataset) -> osgeo.gdal.Dataset:
    """
    Rounds the values of the dataset to the nearest millimeter
    
    :param gdal_file: gdal dataset (dsm or dtm)
    :return: gdal dataset with rounded height values
    """
    array = to_ndarray(gdal_file)
    array = nda_round_to_mm(array)
    return to_gdal_like(array, gdal_file)


def to_wavefront(gdal_file: osgeo.gdal.Dataset, file_path) -> str:
    origin = get_origin(gdal_file)
    pixel_size = get_pixel_size(gdal_file)
    print(f'origin: {origin}')
    print(f'pixel_size: {pixel_size}')
    return nda_to_wavefront(to_ndarray(gdal_file), file_path, origin, pixel_size)


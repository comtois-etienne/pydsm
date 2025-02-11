import osgeo
from osgeo import gdal, ogr, osr
import numpy as np
from skimage import draw

from scipy.ndimage import median_filter
from scipy.ndimage import distance_transform_edt
from cv2 import resize, INTER_CUBIC

from .nda import to_gdal as nda_to_gdal
from .nda import round_to_mm as nda_round_to_mm
from .nda import save_to_wavefront as nda_to_wavefront
from .nda import rescale as nda_rescale
from .nda import dsm_extract_mask as nda_dsm_extract_mask
from .shp import get_coords as shp_get_coords
from .shp import reproject as shp_reproject
from .shp import get_epsg as shp_get_epsg


def open_geotiff(path: str) -> osgeo.gdal.Dataset:
    """
    Opens a geotiff file

    :param path: path to the file
    :return: gdal dataset
    """
    return gdal.Open(path)


def save_geotiff(gdal_file: osgeo.gdal.Dataset, path: str) -> None:
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


def get_origin(gdal_file: osgeo.gdal.Dataset) -> tuple[float]:
    """
    Top left corner of the dataset in the coordinate system
    see https://gdal.org/en/stable/tutorials/geotransforms_tut.html

    :param gdal_file: gdal dataset
    :return: origin of the dataset (x, y) in the coordinate system
    """
    return gdal_file.GetGeoTransform()[0], gdal_file.GetGeoTransform()[3]


def get_center(gdal_file: osgeo.gdal.Dataset) -> tuple[float]:
    """
    Center of the dataset in the coordinate system

    :param gdal_file: gdal dataset
    :return: center of the dataset (x, y) in the coordinate system
    """
    shape = get_shape(gdal_file)
    return get_coordinate_at_pixel(gdal_file, (shape[0] // 2, shape[1] // 2))


def get_scales(gdal_file: osgeo.gdal.Dataset) -> tuple[float]:
    """
    Spacial resolution of the dataset in the coordinate system
    West-East pixel resolution, North-South pixel resolution
    North-South pixel resolution is negative
    see https://gdal.org/en/stable/tutorials/geotransforms_tut.html

    :param gdal_file: gdal dataset
    :return: dimension of the pixel in the coordinate system (m/px, -m/px)
    """
    return gdal_file.GetGeoTransform()[1], gdal_file.GetGeoTransform()[5]


def get_shape(gdal_file: osgeo.gdal.Dataset) -> tuple[int]:
    """
    Pixel sizes of the dataset

    :param gdal_file: gdal dataset
    :return: shape of the dataset (height, width) or (y, x)
    """
    return gdal_file.RasterYSize, gdal_file.RasterXSize


def get_size(gdal_file: osgeo.gdal.Dataset) -> tuple[float]:
    """
    Spacial size of the dataset in meters

    :param gdal_file: gdal dataset
    :return: size of the dataset in meters (width, height) or (x, y)
    """
    y, x = get_shape(gdal_file)
    pixel_size = get_scales(gdal_file)
    return abs(x * pixel_size[0]), abs(y * pixel_size[1])


def get_dtype(gdal_file: osgeo.gdal.Dataset):
    """
    :param gdal_file: gdal dataset
    :return: data type of the dataset
    """
    return gdal_file.GetRasterBand(1).DataType


# COORDINATES

def get_coordinate_at_pixel(gdal_file: osgeo.gdal.Dataset, px: tuple[int], precision=3) -> tuple[float]:
    """
    :param gdal_file: gdal dataset
    :param px: pixel of the coordinate (i, j) or (y, x) in the dataset
    :param precision: number of decimals to round the coordinate (default: 3 (millimetric precision))
    :return: coordinate of the pixel (x, y) in the coordinate system
    """
    y, x = px
    origin = get_origin(gdal_file)
    pixel_size = get_scales(gdal_file)
    h, w = get_shape(gdal_file)
    x = w - x if x < 0 else x
    y = h - y if y < 0 else y
    return round(origin[0] + x * pixel_size[0], precision), round(origin[1] + y * pixel_size[1], precision)


def get_coordinates_at_pixels(gdal_file: osgeo.gdal.Dataset) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :return: array of the coordinates of the pixels in the coordinate system (x, y)
    """
    origin = get_origin(gdal_file)
    pixel_size = get_scales(gdal_file)
    x = np.arange(gdal_file.RasterXSize)
    y = np.arange(gdal_file.RasterYSize)
    xx, yy = np.meshgrid(x, y)
    all_x = origin[0] + xx * pixel_size[0]
    all_y = origin[1] + yy * pixel_size[1]
    return np.stack((all_x, all_y), axis=-1)


def get_pixel_at_coordinate(gdal_file: osgeo.gdal.Dataset, xy: tuple[float]) -> tuple[int]:
    """
    :param gdal_file: gdal dataset
    :param xy: coordinate of the pixel (x, y) in the coordinate system
    :return: pixel of the coordinate (i, j) or (y, x) in the dataset
    """
    origin = get_origin(gdal_file)
    pixel_size = get_scales(gdal_file)
    return int((xy[1] - origin[1]) / pixel_size[1]), int((xy[0] - origin[0]) / pixel_size[0])


def get_pixels_at_coordinates(gdal_file: osgeo.gdal.Dataset, coords: np.ndarray | list) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :param coords: array of the coordinates in the coordinate system (x, y)
    :return: array of the pixels positions in the dataset (i, j) or (y, x)
    """
    coords = np.array(coords)
    origin = get_origin(gdal_file)
    pixel_size = get_scales(gdal_file)
    pixels = np.stack((
        (coords[:, 1] - origin[1]) / pixel_size[1],
        (coords[:, 0] - origin[0]) / pixel_size[0] 
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


def to_ndarray(gdal_file: osgeo.gdal.Dataset, band_count=None) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :param band_count: number of bands to read (1 for grayscale, 4 for RGB with alpha)
    :return: array of the dataset
    """
    band_count = band_count if band_count else gdal_file.RasterCount
    if band_count > 1:
        return np.dstack([gdal_file.GetRasterBand(i).ReadAsArray() for i in range(1, band_count + 1)])
    return gdal_file.GetRasterBand(1).ReadAsArray()


def to_xyz(gdal_file: osgeo.gdal.Dataset) -> np.ndarray:
    """
    :param gdal_file: gdal dataset (dsm like)
    :return: array of the coordinates of the pixels in the coordinate system (x, y, z)
    """
    coordinates = get_coordinates_at_pixels(gdal_file) # x, y
    nda = to_ndarray(gdal_file) # z
    return np.dstack((coordinates, nda))


def to_gdal_like(nda: np.ndarray, gdal_like: gdal.Dataset) -> gdal.Dataset:
    """
    Converts a NumPy array to a GDAL dataset with the same metadata as the input dataset.
    
    :param nda: NumPy array (2D or 3D)
    :param gdal_like: GDAL dataset to copy metadata from
    :return: GDAL dataset
    """
    pixel_size = get_scales(gdal_like)[0]
    origin = get_origin(gdal_like)
    epsg = get_epsg(gdal_like)
    return nda_to_gdal(nda, epsg, origin, pixel_size)


def rescale(gdal_file: osgeo.gdal.Dataset, scale: float) -> osgeo.gdal.Dataset:
    """
    Rescales the dataset to the given scale in meters per pixel

    :param gdal_file: gdal dataset
    :param scale: new scale in meters per pixel
    :return: gdal dataset with the new scale
    """
    current_scales = get_scales(gdal_file)
    array = to_ndarray(gdal_file)
    array = nda_rescale(array, current_scales, scale)
    return nda_to_gdal(array, get_epsg(gdal_file), get_origin(gdal_file), abs(scale))


def correct_dtm(dtm_gdal: osgeo.gdal.Dataset, subsampling_size=500, median_kernel_size=5) -> osgeo.gdal.Dataset:
    """
    Corrects the DTM by filling the gaps and smoothing the surface
    :param dtm_gdal: gdal dataset of the DTM
    :param subsampling_size: size of the subsampling (default: 500)
    :param median_kernel_size: size of the median kernel (default: 5)
    :return: corrected gdal dataset of the DTM
    """
    dtm_array = to_ndarray(dtm_gdal)
    dtm_array, mask = nda_dsm_extract_mask(dtm_array)
    mask = mask.astype(bool)
    shape = dtm_array.shape

    _, nearest_indices = distance_transform_edt(~mask, return_indices=True)
    filled_dtm = dtm_array[tuple(nearest_indices)]

    dtm_simple = filled_dtm[::subsampling_size, ::subsampling_size]
    dtm_smoothed = median_filter(dtm_simple, size=median_kernel_size)

    dtm_upsampled = resize(dtm_smoothed, (shape[1], shape[0]), interpolation=INTER_CUBIC)
    dtm_upsampled[~mask] = -9999.0

    dtm_corrected = to_gdal_like(dtm_upsampled, dtm_gdal)
    return dtm_corrected


def to_ndsm(dsm_gdal: osgeo.gdal.Dataset, dtm_gdal: osgeo.gdal.Dataset, capture_height=120.0, round_to_millimeters=True) -> np.ndarray:
    """
    :param dsm_gdal: gdal dataset of the DSM
    :param dtm_gdal: gdal dataset of the DTM
    :param capture_height: height of the capture device in meters - used to remove values above the drone (default: 120.0)
    :param round_to_millimeters: round the values to the nearest millimeter (default: True)
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
    ndsm[ndsm > float(capture_height)] = 0.0
    ndsm = nda_round_to_mm(ndsm) if round_to_millimeters else ndsm
    ndsm = to_gdal_like(ndsm, dsm_gdal)
    return ndsm


def mask_from_coords(gdal_file: osgeo.gdal.Dataset, points: np.ndarray | list) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :param points: array of the positions of the pixels (y, x) or (i, j)
    :return: mask of the coordinates in the dataset
    """
    points = np.array(points)
    mask = np.zeros(get_shape(gdal_file), dtype=bool)
    r = [point[0] for point in points]
    c = [point[1] for point in points]
    rr, cc = draw.polygon(r, c)
    mask[rr, cc] = True
    return mask


def mask_from_shapefile(gdal_file: osgeo.gdal.Dataset, shapefile_path: str) -> np.ndarray:
    """
    :param gdal_file: gdal dataset
    :param shapefile_path: path to the shapefile
    :return: mask of the shapefile in the dataset
    """
    coords = np.array(shp_get_coords(shapefile_path))
    points = get_pixels_at_coordinates(gdal_file, coords)
    return mask_from_coords(gdal_file, points)


def crop_from_shapefile(gdal_file: osgeo.gdal.Dataset, shapefile_path: str, mask_value=0.0) -> osgeo.gdal.Dataset:
    """
    :param gdal_file: gdal dataset
    :param shapefile_path: path to the shapefile
    :param mask_value: value to fill the mask
    :return: cropped gdal dataset from the shapefile
    """
    gdal_epsg = get_epsg(gdal_file)
    coords = np.array(shp_get_coords(shapefile_path))
    coords = np.array(shp_reproject(coords, shp_get_epsg(shapefile_path), gdal_epsg))
    points = get_pixels_at_coordinates(gdal_file, coords)
    mask = mask_from_coords(gdal_file, points)
    array = to_ndarray(gdal_file)
    mask = mask if len(array.shape) == 2 else np.stack([mask for _ in range(array.shape[2])], axis=2)
    array = array * mask
    array[~mask] = mask_value
    min_py, min_px = np.min(points, axis=0)
    max_py, max_px = np.max(points, axis=0)
    array = array[min_py:max_py, min_px:max_px]

    xy = get_coordinate_at_pixel(gdal_file, (min_py, min_px))
    gdal_croped = nda_to_gdal(
        array, 
        gdal_epsg, 
        xy, 
        get_scales(gdal_file)[0]
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
    """
    #todo
    #works at small scale
    """
    origin = get_origin(gdal_file)
    pixel_size = get_scales(gdal_file)
    print(f'origin: {origin}')
    print(f'pixel_size: {pixel_size}')
    return nda_to_wavefront(to_ndarray(gdal_file), file_path, origin, pixel_size)


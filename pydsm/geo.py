import osgeo
from osgeo import gdal, ogr, osr
import numpy as np
from skimage import draw

from scipy.ndimage import median_filter
from scipy.ndimage import distance_transform_edt
from cv2 import resize as cv2_resize
from cv2 import INTER_CUBIC
import cv2

from .nda import to_gdal as nda_to_gdal
from .nda import round_to_mm as nda_round_to_mm
from .nda import save_to_wavefront as nda_to_wavefront
from .nda import rescale as nda_rescale
from .nda import dsm_extract_mask as nda_dsm_extract_mask
from .nda import downsample as nda_downsample

from .shp import open_shapefile as shp_read_coords
from .shp import reproject as shp_reproject
from .shp import get_epsg as shp_get_epsg
from .shp import dilate as shp_dilate
from .shp import graph_from_coord as shp_graph_from_coord
from .shp import get_sample_points as shp_get_sample_points
from .shp import get_surrounding_streets as shp_get_surrounding_streets
from .shp import is_inside as shp_is_inside
from .shp import save_surrounding_streets as shp_save_surrounding_streets
from .shp import plot_path as shp_plot_path
from .shp import area as shp_area

from .utils import *


########## GEOTIF IO ##########

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


########## PROPRIETIES ##########

def get_epsg(gdal_file: osgeo.gdal.Dataset) -> EPSG:
    """
    :param gdal_file: gdal dataset
    :return: EPSG code of the dataset
    """
    proj = osr.SpatialReference(wkt=gdal_file.GetProjection())
    epsg: osgeo.osr.SpatialReference = proj.GetAttrValue('AUTHORITY', 1)
    return int(epsg)


def get_origin(gdal_file: osgeo.gdal.Dataset) -> Coordinate:
    """
    Top left corner of the dataset in the coordinate system  
    see https://gdal.org/en/stable/tutorials/geotransforms_tut.html

    :param gdal_file: gdal dataset
    :return: origin of the dataset (x, y) in the coordinate system
    """
    return gdal_file.GetGeoTransform()[0], gdal_file.GetGeoTransform()[3]


def get_center(gdal_file: osgeo.gdal.Dataset) -> Coordinate:
    """
    Center of the dataset in the coordinate system

    :param gdal_file: gdal dataset
    :return: center of the dataset (x, y) in the coordinate system
    """
    shape = get_shape(gdal_file)
    return get_coordinate_at_pixel(gdal_file, (shape[0] // 2, shape[1] // 2))


def get_scales(gdal_file: osgeo.gdal.Dataset) -> tuple[Scale, Scale]:
    """
    Spacial resolution of the dataset in the coordinate system  
    West-East pixel resolution, North-South pixel resolution  
    North-South pixel resolution is negative  
    see https://gdal.org/en/stable/tutorials/geotransforms_tut.html  

    :param gdal_file: gdal dataset
    :return: dimension of the pixel in the coordinate system (m/px, -m/px)
    """
    return gdal_file.GetGeoTransform()[1], gdal_file.GetGeoTransform()[5]


def get_shape(gdal_file: osgeo.gdal.Dataset) -> Shape:
    """
    Pixel sizes of the dataset

    :param gdal_file: gdal dataset
    :return: shape of the dataset (height, width) or (y, x)
    """
    return gdal_file.RasterYSize, gdal_file.RasterXSize


def get_size(gdal_file: osgeo.gdal.Dataset) -> Size:
    """
    Spacial size of the dataset in meters

    :param gdal_file: gdal dataset
    :return: size of the dataset in meters (width, height) or (x, y)
    """
    y, x = get_shape(gdal_file)
    pixel_size = get_scales(gdal_file)
    return abs(x * pixel_size[0]), abs(y * pixel_size[1])


def get_dtype(gdal_file: osgeo.gdal.Dataset) -> int:
    """
    :param gdal_file: gdal dataset
    :return: data type of the dataset
    """
    return gdal_file.GetRasterBand(1).DataType


def get_bbox(gdal_file: osgeo.gdal.Dataset, format_coordinates='bbox') -> Coordinates:
    """
    :param gdal_file: gdal dataset
    :param format_coordinates: format of the bounding box ('bbox' or 'polygon')  
        - `bbox` bounding box of the dataset ( (min_x, min_y), (max_x, max_y) )  
        - `polygon` bounding box of the dataset as a polygon ( 4 coordinates )  
        - `ring` bounding box of the dataset as a closed ring ( 5 coordinates )  
    :return: bounding box of the dataset ( (min_x, min_y), (max_x, max_y) )
    """
    origin = get_origin(gdal_file)
    size = get_size(gdal_file)
    bbox = [ origin, (origin[0] + size[0], origin[1] - size[1]) ]
    polygon = [ bbox[0], (bbox[1][0], bbox[0][1]), bbox[1], (bbox[0][0], bbox[1][1])]
    ring = polygon + [polygon[0]]

    if format_coordinates == 'bbox':
        return bbox
    if format_coordinates == 'polygon':
        return polygon
    if format_coordinates == 'ring':
        return ring
    raise ValueError("format_coordinates must be 'bbox', 'polygon' or 'ring'")


########## COORDINATES ##########

def get_coordinate_at_pixel(gdal_file: osgeo.gdal.Dataset, point: Point, precision=3) -> Coordinate:
    """
    :param gdal_file: gdal dataset
    :param point: pixel of the desired coordinate (i, j) or (y, x) in the dataset
    :param precision: number of decimals to round the coordinate (default: 3 (millimetric precision))
    :return: coordinate of the pixel (x, y) in the coordinate system
    """
    y, x = point
    origin = get_origin(gdal_file)
    pixel_size = get_scales(gdal_file)
    h, w = get_shape(gdal_file)
    x = w - x if x < 0 else x
    y = h - y if y < 0 else y
    return round(origin[0] + x * pixel_size[0], precision), round(origin[1] + y * pixel_size[1], precision)


def get_coordinates_at_pixels(gdal_file: osgeo.gdal.Dataset) -> Coordinates | np.ndarray:
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


def get_pixel_at_coordinate(gdal_file: osgeo.gdal.Dataset, xy: Coordinate) -> Point:
    """
    :param gdal_file: gdal dataset
    :param xy: coordinate of the pixel (x, y) in the coordinate system
    :return: pixel of the coordinate (i, j) or (y, x) in the dataset
    """
    origin = get_origin(gdal_file)
    pixel_size = get_scales(gdal_file)
    return int((xy[1] - origin[1]) / pixel_size[1]), int((xy[0] - origin[0]) / pixel_size[0])


def get_pixels_at_coordinates(gdal_file: osgeo.gdal.Dataset, coords: Coordinates) -> Points:
    """
    :param gdal_file: gdal dataset
    :param coords: array of the coordinates in the coordinate system (x, y) or (lon, lat)
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


########## CONVERSIONS ##########

def reproject(gdal_file: osgeo.gdal.Dataset, epsg: EPSG) -> osgeo.gdal.Dataset:
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


def to_ndarray(gdal_file: osgeo.gdal.Dataset, band_count: int = None) -> np.ndarray:
    """
    Extracts the array of the dataset  
    Uses gdal_file.RasterCount to get the number of bands if band_count is not provided

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
    Creates a matrix containing every coordinate and its altitude  
    `Warning`: the resulting matrix is very large and should be used with caution  
    Use get_coordinates_at_pixel for a more efficient way to get the coordinates

    :param gdal_file: gdal dataset (dsm like - no orthophoto)
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


def round_to_mm(gdal_file: osgeo.gdal.Dataset) -> osgeo.gdal.Dataset:
    """
    Rounds the values of the dataset to the nearest millimeter
    
    :param gdal_file: gdal dataset (dsm or dtm)
    :return: gdal dataset with rounded height values
    """
    array = to_ndarray(gdal_file)
    array = nda_round_to_mm(array)
    return to_gdal_like(array, gdal_file)


def rescale(gdal_file: osgeo.gdal.Dataset, scale: Scale) -> osgeo.gdal.Dataset:
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


def resize(gdal_file: osgeo.gdal.Dataset, shape: Shape, scale: Scale = None) -> osgeo.gdal.Dataset:
    """
    Resizes the dataset to the given shape  
    `Warning`: the scale of the dataset is not preserved.  
    This should be used for small changes in the shape.  

    :param gdal_file: gdal dataset
    :param shape: new shape (height, width) or (y, x)
    :return: gdal dataset with the new shape
    """
    scale = scale or abs(get_scales(gdal_file)[0])
    shape = (shape[1], shape[0])
    array = to_ndarray(gdal_file)
    array = cv2_resize(array, shape, interpolation=INTER_CUBIC)
    return nda_to_gdal(array, get_epsg(gdal_file), get_origin(gdal_file), scale)


def resize_like(gdal_file: osgeo.gdal.Dataset, gdal_like: osgeo.gdal.Dataset) -> osgeo.gdal.Dataset:
    """
    Resizes the dataset to the shape of the given dataset  
    `Warning`: the scale and shape of the gdal_like dataset is used.  
    This should be used for small changes in the shape or if you are certain that the scales are the same.  

    :param gdal_file: gdal dataset
    :param gdal_like: gdal dataset to copy the shape and scale from
    :return: gdal dataset with the new shape
    """
    shape = get_shape(gdal_like)
    scale = abs(get_scales(gdal_like)[0])
    return resize(gdal_file, shape, scale)


def translation(gdal_file: osgeo.gdal.Dataset, translate: Coordinate) -> osgeo.gdal.Dataset:
    """
    2D translation of the dataset

    :param gdal_file: gdal dataset
    :param translate: translation vector (x, y) from the origin (top-left corner)
    :return: gdal dataset translated
    """
    array = to_ndarray(gdal_file)
    origin = get_origin(gdal_file)
    pixel_size = get_scales(gdal_file)
    new_origin = (origin[0] + translate[0], origin[1] + translate[1])
    return nda_to_gdal(array, get_epsg(gdal_file), new_origin, abs(pixel_size[0]))


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

    dtm_upsampled = cv2_resize(dtm_smoothed, (shape[1], shape[0]), interpolation=INTER_CUBIC)
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


def mask_from_points(gdal_file: osgeo.gdal.Dataset, points: Points) -> np.ndarray:
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
    coords = np.array(shp_read_coords(shapefile_path))
    points = get_pixels_at_coordinates(gdal_file, coords)
    return mask_from_points(gdal_file, points)


def crop_from_shapefile(gdal_file: osgeo.gdal.Dataset, shapefile_path: str, mask_value=0.0, dilate_size=0.0) -> osgeo.gdal.Dataset:
    """
    :param gdal_file: gdal dataset
    :param shapefile_path: path to the shapefile (.shp)
    :param mask_value: value to fill the mask
    :param dilate_size: size of the dilation in meters
    :return: cropped gdal dataset from the shapefile
    """
    gdal_epsg = get_epsg(gdal_file)
    coords = np.array(shp_read_coords(shapefile_path))
    coords = np.array(shp_reproject(coords, shp_get_epsg(shapefile_path), gdal_epsg))
    if dilate_size > 0.0: coords = np.array(shp_dilate(coords, dilate_size))
    points = get_pixels_at_coordinates(gdal_file, coords)

    mask = mask_from_points(gdal_file, points)
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


def extract_zones(geotif_path: str, save_directory: str = './', street_name_exclusions: list[str] = None, search_radius: int=500, sample_size: int=3, safe_zone: float=10, min_area=2500) -> list[UUIDv4]:
    """
    Extracts the zones around the points of the geotiff file that are inside the bounding box of the geotiff
    Saves the shapefiles of the extracted zones in the save_directory

    :param path: path to the geotiff file
    :param save_directory: directory to save the shapefiles (default './')
    :param street_name_exclusions: list of words to exclude from the street names
    :param search_radius: radius around the point to search for streets (default 500 meters)
    :param sample_size: number of points on the longest side to use as seed for the path search
    :param safe_zone: disntance around the extracted path that needs to be inside the bounding box of the geotiff
    :param min_area: minimum area of the zone in square meters
    :return: list of uuid strings of the saved shapefiles
    """
    gdal_file = open_geotiff(geotif_path)

    epsg = get_epsg(gdal_file)
    origin = get_origin(gdal_file)
    origin = shp_reproject([origin], epsg, CRS_GPS)[0]
    G = shp_graph_from_coord(origin, search_radius)

    bbox_src = get_bbox(gdal_file, format_coordinates='ring')

    points = shp_get_sample_points(get_shape(gdal_file), sample_size, True)
    coordinates = [get_coordinate_at_pixel(gdal_file, point) for point in points]
    coordinates = shp_reproject(coordinates, epsg, CRS_GPS)

    zones = {} # used to eliminate duplicates
    for coord in coordinates:
        res = shp_get_surrounding_streets(G, coord, street_name_exclusions)
        if res is None: continue

        uuid_str, path_coords, indexes = res
        path_coords_dst = shp_reproject(path_coords, CRS_GPS, epsg)
        dilate_path = shp_dilate(path_coords_dst, safe_zone)

        area = shp_area(dilate_path)
        if shp_is_inside(bbox_src, dilate_path) and area > min_area:
            zones[uuid_str] = (path_coords, indexes)
            # shp_plot_path(path_coords, coord)

    for uuid_str, (path_coords, indexes) in zones.items():
        shp_save_surrounding_streets(uuid_str, path_coords, indexes, save_directory)

    return list(zones.keys())


def get_coordinate_on_click(gdal_file: osgeo.gdal.Dataset, downsample: int = 20) -> Coordinate:
    """
    Selects a point on the geotiff by clicking on it to get its coordinate
    `Warning`: use exit() to prevent cv2 to continue running on macos

    :param gdal_file: gdal dataset
    :param downsample: downsample factor for the display (default 20)
    :return: coordinate at the selected point
    """
    title = 'geotiff (close window when point is selected)'
    array = to_ndarray(gdal_file)
    array = nda_downsample(array, downsample)
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)    
    canvas = array.copy()
    point = None

    def click_event(event, x, y, flags, param):
        nonlocal point
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)
            canvas = array.copy()
            cv2.circle(canvas, point, 5, (0, 0, 255), -1)
            cv2.imshow(title, canvas)

    cv2.imshow(title, canvas)
    cv2.setMouseCallback(title, click_event)

    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    point = (point[1] * downsample, point[0] * downsample)
    coord = get_coordinate_at_pixel(gdal_file, point)

    return coord


def to_wavefront(gdal_file: osgeo.gdal.Dataset, file_path: str):
    """
    #todo
    #works at small scale
    """
    origin = get_origin(gdal_file)
    pixel_size = get_scales(gdal_file)
    print(f'origin: {origin}')
    print(f'pixel_size: {pixel_size}')
    nda_to_wavefront(to_ndarray(gdal_file), file_path, origin, pixel_size)


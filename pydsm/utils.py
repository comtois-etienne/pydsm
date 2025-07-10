import numpy as np
from osgeo import gdal
import json
import os


########## TYPES ##########

DTYPE_TO_GDAL = {
    int: gdal.GDT_Byte, # 1
    float: gdal.GDT_Float64, # 7
    np.int64: gdal.GDT_Int64, # 13
    np.uint8: gdal.GDT_Byte, # 1
    np.float32: gdal.GDT_Float32, # 6
    np.float64: gdal.GDT_Float64, # 7
}
"""
Mapping of Python data types to GDAL data types  
"""


DTYPE_TO_NP = {
    gdal.GDT_Byte: np.uint8,
    gdal.GDT_Float32: np.float32,
    gdal.GDT_Float64: np.float64,
    gdal.GDT_Int64: np.int64,
}
"""
Mapping of GDAL data types to NumPy data types  
"""


Point = tuple[int, int]
"""
`(y, x)` or `(i, j)`  
A point (pixel) on a matrix  
Origin is at the top-left corner of the matrix  
The first element of the tuple is the row index (y) or (i)  
The second element of the tuple is the column index (x) or (j)  
"""


Points = list[Point]
"""
`[ (y, x), ... ]`  or `[ (i, j), ... ]`
A list of points (pixels) on a matrix 
"""


Coordinate = tuple[float, float] | tuple[int, int]
"""
`(lon, lat)` or `(x, y)`  
A coordinate on the Earth's surface or a Cartesian coordinate   
The first element of the tuple is the longitude (lon) or x-coordinate in meters  
The second element of the tuple is the latitude (lat) or y-coordinate in meters  
"""


Coordinates = list[Coordinate]
"""
`[ (lon, lat), ... ]` or `[ (x, y), ... ]`  
A list of coordinates on the Earth's surface or Cartesian coordinates  
"""


Meters = float
"""
Distance in meters
"""


Shape = tuple[int, int]
"""
`(height, width)` or `(y, x)`  
Shape of a matrix (height x width) in pixels
"""


Size = tuple[Meters, Meters]
"""
`(width, height)` or `(x, y)`  
Size of a matrix (width x height) in meters
"""


Scale = float
"""
`m / px`  
Scale of a matrix (meters per pixel)
"""


#todo rename NoneIndex
NodeIndex = int
"""
Node index (street intersection id)
"""

#todo rename NoneIndexes
NodeIndexes = list[NodeIndex]
"""
A list of node indexes (street intersection ids) [index_0, ...]
"""


UUIDv4 = str
"""
UUIDv4 string - a unique identifier for a path (hash of the path)
"""


EPSG = int
"""
EPSG code of a coordinate system
"""


########## CONSTANTS ##########

CRS_GPS = 4326
"""
Coordinate Reference System (CRS) for GPS coordinates (WGS84)
"""


CRS_CAN = 2950
"""
Coordinate Reference System (CRS) for Cartesian coordinates (NAD83(CSRS))
"""


########## PATH FUNCTIONS ##########

def append_file_to_path(directory: str, filename: str) -> str:
    """
    Format the full path to save the file

    :param directory: Directory path to save the file
    :param filename: File name in the directory
    :return: Full path to save the file
    """
    if directory == '':
        return filename
    directory = directory[:-1] if directory[-1] == '/' else directory
    return f"{directory}/{filename}"


def get_folder_path(filepath: str) -> str:
    """
    Get the folder path from the file path

    :param filepath: File path (example: /path/to/file.txt)
    :return: Folder path (example: /path/to)
    """
    return '/'.join(filepath.split('/')[:-1])


def get_filename(filepath: str) -> str:
    """
    Get the file name from the file path

    :param filepath: File path (example: /path/to/file.txt)
    :return: File name (example: file.txt)
    """
    return filepath.split('/')[-1]


def get_extension(filepath: str) -> str:
    """
    Get the file extension from the filename

    :param filename: File name with extension (example: /path/to/file.txt)
    :return: File extension (example: txt)
    """
    filename = get_filename(filepath)
    split = filename.split('.')
    if len(split) == 1:
        return ''
    return split[-1]


def remove_extension(filepath: str) -> str:
    """
    Remove the file extension from the filename

    :param filename: File name with extension (example: /path/to/file.txt)
    :return: File name without extension (example: /path/to/file)
    """
    ext = get_extension(filepath)
    if ext == '':
        return filepath
    return filepath[:-len(ext)-1]


########## EPSG FUNCTIONS ##########

def epsgio_link_from_coord(coord: Coordinate, epsg: EPSG, zoom: int = 18, layer="osm") -> str:
    """
    Generate a link to epsg.io map with the given coordinate  

    :param coord: tuple of (x, y) coordinate
    :param epsg: int, EPSG code of the coordinate system
    :param zoom: int, zoom level of the map
    :param layer: str, layer of the map (osm, streets, satellite)
    :return: str, link to the map
    """
    # format the coordinate to have 6 decimal places with trailing zeros
    x = "{:.6f}".format(coord[0])
    y = "{:.6f}".format(coord[1])
    return f"https://epsg.io/map#srs={epsg}&x={x}&y={y}&z={zoom}&layer={layer}"


def epsgio_link_to_coord(url: str) -> Coordinate:
    """
    Extract the coordinate from the epsg.io link

    :param link: str, link to the map
    :return: tuple of (x, y) coordinate
    """
    url = url.split("#")[1]
    url = url.split("&")
    coord = [float(url[1].split("=")[1]), float(url[2].split("=")[1])]
    return tuple(coord)


########## JSON FUNCTIONS ##########

def write_dict_as_json(metadata: dict, file_path: str) -> None:
    """
    Write the dict to a json file.

    :param metadata: Metadata to write
    :param file_path: Name of the json file
    """
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4)


########## CONVERSIONS FUNCTIONS ##########

def pixel_to_height(pixel: int, scale: Scale, padding: int=0) -> Meters:
    """
    Convert pixel to height in meters.

    :param pixel: Pixel value at the specified scale
    :param padding: Padding to remove from the pixel value
    :param scale: Scale of the image in meters per pixel
    :return: Height in meters
    """
    return (pixel - padding) * scale


def height_to_pixel(height: Meters, scale: Scale, padding: int=0) -> int:
    """
    Convert height in meters to pixel.

    :param height: Height in meters
    :param padding: Padding to add to the pixel value
    :param scale: Scale of the image in meters per pixel
    :return: Pixel value
    """
    return int(height / scale) + padding


########## POINTS FUNCTIONS ##########

def rotate_points(center: tuple[float, float], points: np.array, angle_deg: float) -> np.ndarray:
    """
    Rotate points around a center by a given angle in degrees.

    :param center: Tuple of the center coordinates (y, x)
    :param points: 2D numpy array of the points to rotate (y, x, z)
    :param angle_deg: Angle in degrees to rotate the points
    :return: 2D numpy array of the rotated points (y, x, z)
    """
    angle_rad = np.deg2rad(angle_deg)
    cy, cx = center[0], center[1]
    y, x, z = points[:, 0], points[:, 1], points[:, 2]

    dy, dx = y - cy, x - cx
    ry = dy * np.cos(angle_rad) - dx * np.sin(angle_rad)
    rx = dy * np.sin(angle_rad) + dx * np.cos(angle_rad)
    ny, nx = ry + cy, rx + cx

    rotated = np.stack((ny, nx, z), axis=1)
    return rotated


def add_z_to_points(points: list, z: int | float) -> np.ndarray:
    """
    Adds the z value to the points the coordinates of the points are kept in the same order.
    todo : optimize this function with numpy
    
    :param points: 2D numpy array of the points
    :param z: Z value to add
    :return: 2D numpy array with the z value added (x, y, z)
    """
    return np.array([[x, y, z] for x, y in points])


def distance_to_point(coords: Points, point: Point, to_int=True):
    """
    Calculate the distance from a point to a set of coordinates.
    :param coords: 2D numpy array of coordinates [(y, x), ...]
    :param point: Point (y, x) to calculate the distance to
    :param to_int: If True, return the distances as integers (default is True)
    :return: 1D numpy array of distances from the point to each coordinate
    """
    coords = np.array(coords)
    distances = np.sqrt((coords[:, 0] - point[0])**2 + (coords[:, 1] - point[1])**2)
    return distances.astype(int) if to_int else distances


########## DRONE SURVEY FUNCTIONS ##########

def get_capture_airspeed(diagonal_fov, image_ratio, hagl, time_delay=2.0, minimum_overlap=0.7) -> float:
    """
    Calculate the drone capture airspeed based on the diagonal field of view, image ratio, and height above ground level (HAGL).  
    Requirements for DJI Mini 3 Pro with WebODM:
    - fov=82.1 degrees (24mm lens)
    - image_ratio=4/3
    - hagl=60.0 meters
    - time_delay=2.0 seconds (default)
    - minimum_overlap=0.7 (default)

    :param diagonal_fov: Diagonal field of view in degrees.
    :param image_ratio: Aspect ratio of the image (width / height).
    :param hagl: Height above ground level in meters.
    :param time_delay: Time delay between each photos in seconds (default is 2.0).
    :param minimum_overlap: Minimum overlap between images as a fraction (default is 0.7).
    :return: Capture airspeed in m/s.
    """
    diagonal_length = 2 * hagl * np.tan(np.radians(diagonal_fov / 2))
    vertical_lenght = diagonal_length / np.sqrt(1 + image_ratio**2)
    overlap_distance = vertical_lenght * (1 - minimum_overlap)
    capture_airspeed = overlap_distance / time_delay
    return capture_airspeed


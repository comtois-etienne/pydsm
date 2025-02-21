import numpy as np
from osgeo import gdal


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


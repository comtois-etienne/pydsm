

########## TYPES ##########

Point = tuple[int, int]
"""
A point (pixel) on a matrix
Origin is at the top-left corner of the matrix
The first element of the tuple is the row index (y) or (i)
The second element of the tuple is the column index (x) or (j)
"""


Points = list[Point]
"""
A list of points (pixels) on a matrix [ (i, j), ... ] or [ (y, x), ... ]
"""


Coordinate = tuple[float, float] | tuple[int, int]
"""
A coordinate on the Earth's surface or a Cartesian coordinate
The first element of the tuple is the longitude (lon) or x-coordinate
The second element of the tuple is the latitude (lat) or y-coordinate
"""


Coordinates = list[Coordinate]
"""
A list of coordinates on the Earth's surface or Cartesian coordinates
[ (lon, lat), ... ] or [ (x, y), ... ]
"""


Index = int
"""
Node index (street intersection id)
"""


Indexes = list[Index]
"""
A list of node indexes (street intersection ids) [index_0, ...]
"""


UUIDv4 = str
"""
UUIDv4 string - a unique identifier for a path (hash of the path)
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


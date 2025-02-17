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



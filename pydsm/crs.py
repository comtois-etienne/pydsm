Point = tuple[int, int] # (i, j) or (y, x) cartesian coordinates of a pixel
Points = list[Point] # list of POINT
Coordinate = tuple[float, float] | tuple[int, int] # (lon, lat) or (x, y)
Coordinates = list[Coordinate] # list of coordinates [(lon, lat), ...]
Index = int # node index (street intersection id)
Indexes = list[Index] # list of node indexes
UUIDv4 = str # unique identifier for a path (hash of the path)

CRS_GPS = 4326


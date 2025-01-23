from osgeo import gdal, ogr, osr


def open(path: str):
    """
    Opens a Shapefile

    :param path: Path to the Shapefile
    :return: The opened Shapefile
    """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("ESRI Shapefile driver not available.")
    
    shapefile = driver.CreateDataSource(path)
    if shapefile is None:
        raise RuntimeError("Could not create shapefile.")
    
    return shapefile


def from_gdal(gdal_file: gdal.Dataset, shapefile_path: str) -> None:
    """
    Extracts non-zero areas from a GDAL raster dataset and saves them as polygons in a shapefile.
    
    :param gdal_file: GDAL dataset (raster)
    :param shapefile_path: Path to the output shapefile
    """
    band = gdal_file.GetRasterBand(1)
    shapefile = open(shapefile_path)
    
    srs = osr.SpatialReference()
    srs.ImportFromWkt(gdal_file.GetProjection())
    
    layer = shapefile.CreateLayer("layer", srs, ogr.wkbPolygon)
    
    field = ogr.FieldDefn("value", ogr.OFTInteger)
    layer.CreateField(field)
    
    gdal.Polygonize(band, None, layer, 0, [], callback=None)
    
    shapefile = None  # close the shapefile


def from_coords(coords: list, epsg: int, shapefile_path: str) -> None:
    """
    Creates a shapefile from a list of coordinates.
    
    :param coords: List of coordinate tuples [(x1, y1), (x2, y2), ...] for points or [(x1, y1), (x2, y2), ..., (xn, yn)] for a polygon
    :param epsg: EPSG code for the coordinate system
    :param shapefile_path: Path where the shapefile will be saved
    """
    shapefile = open(shapefile_path)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    
    layer = shapefile.CreateLayer("polygon_layer", srs, ogr.wkbPolygon)

    # Add an ID field to the shapefile layer (optional)
    field = ogr.FieldDefn("ID", ogr.OFTInteger)
    layer.CreateField(field)

    # Create a ring for the polygon using the coordinates
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in coords:
        ring.AddPoint(x, y)
    
    # Close the ring if not already closed
    if coords[0] != coords[-1]:
        ring.AddPoint(coords[0][0], coords[0][1])
    
    # Create the polygon geometry
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    
    # Create feature and add to layer
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(polygon)
    feature.SetField("ID", 1)
    layer.CreateFeature(feature)

    # Close
    feature = None
    shapefile = None


def get_coords(shapefile_path: str) -> list:
    """
    Converts a shapefile to a list of coordinates.
    
    :param shapefile_path: Path to the shapefile
    :return: List of coordinates for each feature. If the shapefile contains points, lines, or polygons, it will return a list of lists or tuples depending on geometry.
    """
    # Open the shapefile
    datasource = ogr.Open(shapefile_path)
    if datasource is None:
        raise RuntimeError("Could not open shapefile.")
    
    layer = datasource.GetLayer()
    coords_list = []
    
    for feature in layer:
        geometry = feature.GetGeometryRef()
        
        # # Point geometries
        # if geometry.GetGeometryType() == ogr.wkbPoint:
        #     coords = (geometry.GetX(), geometry.GetY())
        #     coords_list.append(coords)
        
        # # Line geometries
        # elif geometry.GetGeometryType() == ogr.wkbLineString:
        #     line_coords = []
        #     for i in range(geometry.GetPointCount()):
        #         x, y, _ = geometry.GetPoint(i)
        #         line_coords.append((x, y))
        #     coords_list.append(line_coords)
        
        # Polygon geometries
        # elif geometry.GetGeometryType() == ogr.wkbPolygon:
        polygon_coords = []
        for ring in geometry:
            ring_coords = []
            for i in range(ring.GetPointCount()-1):
                x, y, _ = ring.GetPoint(i)
                ring_coords.append([x, y])
            polygon_coords.append(ring_coords)
        coords_list.append(polygon_coords)
    
    datasource = None # Close
    return coords_list[0][0]


def get_epsg(shapefile_path: str) -> int:
    """
    Gets the EPSG code of a shapefile.
    
    :param shapefile_path: Path to the shapefile
    :return: EPSG code
    """
    datasource = ogr.Open(shapefile_path)
    if datasource is None:
        raise RuntimeError("Could not open shapefile.")
    
    layer = datasource.GetLayer()
    srs = layer.GetSpatialRef()
    epsg = srs.GetAuthorityCode(None)
    
    datasource = None  # Close
    return int(epsg)


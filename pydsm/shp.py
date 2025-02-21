from dataclasses import dataclass
import hashlib
import uuid
import time
import math

from .utils import *

from osgeo import gdal, ogr, osr
from shapely.geometry import Point, Polygon
import plotly.express as px
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx


########## SHAPEFILE IO ##########

def __open_shapefile(path: str):
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


def save_from_gdal(gdal_file: gdal.Dataset, shapefile_path: str) -> None:
    """
    Extracts non-zero areas from a GDAL raster dataset and saves them as polygons in a shapefile.
    
    :param gdal_file: GDAL dataset (raster)
    :param shapefile_path: Path to the output shapefile
    """
    band = gdal_file.GetRasterBand(1)
    shapefile = __open_shapefile(shapefile_path)
    
    srs = osr.SpatialReference()
    srs.ImportFromWkt(gdal_file.GetProjection())
    
    layer = shapefile.CreateLayer("layer", srs, ogr.wkbPolygon)
    
    field = ogr.FieldDefn("value", ogr.OFTInteger)
    layer.CreateField(field)
    
    gdal.Polygonize(band, None, layer, 0, [], callback=None)
    
    shapefile = None  # close the shapefile


def save_from_coords(coords: Coordinates, epsg: EPSG, shapefile_path: str) -> None:
    """
    Creates a shapefile from a list of coordinates.
    
    :param coords: List of coordinate tuples [(x1, y1), (x2, y2), ...] for points or [(x1, y1), (x2, y2), ..., (xn, yn)] for a polygon
    :param epsg: EPSG code for the coordinate system
    :param shapefile_path: Path where the shapefile will be saved
    """
    coords = np.array(coords)[:,:2].tolist()
    shapefile = __open_shapefile(shapefile_path)

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


def save_from_csv(csv_path: str, shapefile_path: str, epsg: EPSG = None) -> None:
    """
    Creates a shapefile from a CSV file containing coordinates. (look for geo.save_csv)

    :param csv_path: Path to the CSV file containing coordinates (x, y)
    :param shapefile_path: Path to the output shapefile
    :param epsg: EPSG code for the coordinate system. 
        If None, will use the comment line in the CSV file to find the EPSG code.
        example comment line: #epsg=2950
    """
    bounds = pd.read_csv(csv_path, comment='#')
    metadata = {}
    with open(csv_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                k, v = line.strip()[1:].split(',')
                metadata[k] = v
    if 'epsg' not in metadata and epsg is None:
        raise ValueError("No EPSG code found in the CSV file. Add #epsg=NUMBER to the file.")
    epsg = epsg if epsg is not None else int(metadata['epsg'])
    save_from_coords(bounds.values.tolist(), epsg, shapefile_path)


def save_csv(csv_path: str, coordinates: Coordinates, epsg: EPSG = CRS_GPS, metadata: dict = None) -> None:
    """
    Saves a list of coordinates to a csv file

    :param csv_path: path to the csv file
    :param coordinates: list of coordinates (x, y) or (lon, lat)
    :param metadata: metadata to add to the csv file
    :param epsg: crs of the coordinates
    """
    metadata = metadata or {}
    metadata['epsg'] = epsg

    columns = ['x', 'y']
    df = pd.DataFrame(coordinates, columns=columns)
    df.to_csv(csv_path, index=False)

    with open(csv_path, 'r') as original: data = original.read()
    with open(csv_path, 'w') as modified: 
        for key, value in metadata.items():
            modified.write(f'#{key},{value}\n')
        modified.write(data)


def open_shapefile(shapefile_path: str) -> Coordinates:
    """
    Converts a shapefile to a list of coordinates.
    
    :param shapefile_path: Path to the shapefile
    :return: List of coordinates for each feature.  
        If the shapefile contains points, lines, or polygons;  
        it will return a list of lists or tuples depending on geometry.
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


########## SHAPEFILE FUNCTIONS ##########

def get_epsg(shapefile_path: str) -> EPSG:
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


########## COORDINATE FUNCTIONS ##########

def reproject(coordinates: Coordinates, src_epsg: EPSG, dst_epsg: EPSG, round_to_millimeters=True) -> Coordinates:
    """
    Converts a list of coordinates from one projection system to another

    :param array: List of coordinates to convert (x, y) or (lon, lat). Altitude (z) is ignored.
    :param src_epsg: EPSG code of the source projection system
    :param dst_epsg: EPSG code of the destination projection system
    :param round_to_millimeters: Round the coordinates to 3 decimal places (disabled for EPSG:4326 as destination)
    """
    coordinates = np.array(coordinates)[:,:2]

    if src_epsg == CRS_GPS: 
        coordinates = coordinates[:, [1, 0]]
    if dst_epsg == CRS_GPS:
        round_to_millimeters = False

    src = osr.SpatialReference()
    src.ImportFromEPSG(src_epsg)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(dst_epsg)
    transform = osr.CoordinateTransformation(src, dst)
    reprojected = np.array(transform.TransformPoints(coordinates))

    if round_to_millimeters: reprojected = np.round(reprojected, 3)
    reprojected = reprojected[:, :2]

    if dst_epsg == CRS_GPS: 
        reprojected = reprojected[:, [1, 0]]
    
    return reprojected.tolist()


def dilate(coordinates: Coordinates, distance: float=10.0) -> Coordinates:
    """
    Dillates a list of coordinates forming a polygon by a given distance  
    `warning` EPSG:4326 is not supported for dilation. Please use a projected coordinate system.

    :param coordinates: List of coordinates to dilate (x, y) or (lon, lat)
    :param distance: Distance to dilate the coordinates in meters (default: 10.0m)
    :return: List of dilated coordinates
    """
    coords = np.array(coordinates)
    polygon = Polygon(coords)
    buffer_polygon = polygon.buffer(distance)
    return list(buffer_polygon.exterior.coords)


def area(coordinates: Coordinates) -> float:
    """
    Calculates the area of a polygon formed by a list of coordinates

    :param coordinates: List of coordinates forming a polygon (x, y)
    :return: Area of the polygon in square meters
    """
    polygon = Polygon(coordinates)
    return polygon.area


def is_inside(boundary: pd.DataFrame | Coordinates, coordinates: Coordinates) -> bool:
    """
    Returns True if the coordinate is inside the polygon made by the coordinates

    :param boundary: coordinates of the closed polygon [(x_0, y_0), (x_1, y_1), ..., (x_n, y_n), (x_0, y_0)]
    :param coordinates: list of coordinates (x, y) or (lon, lat) that need to be inside the polygon
    :return: True if all the coordinates are inside the polygon boundary
    """
    if len(coordinates) == 0: return False

    polygon = Polygon(boundary)
    for coordinate in coordinates:
        if not polygon.contains(Point(coordinate)):
            return False
    return True


def plot_path(coordinates: Coordinates, seed_coord: Coordinate) -> None:
    """
    Plot a path on the map

    :param coordinates: list of coordinates [(lon, lat), ...]
    """
    coords = pd.DataFrame(coordinates, columns=['lon', 'lat'])
    coords['index'] = range(len(coords))
    fig = px.line_mapbox(coords, lat='lat', lon='lon', zoom=16, height=900, width=900, hover_data=['index'])
    fig.add_scattermapbox(lat=[seed_coord[1]], lon=[seed_coord[0]], mode='markers', marker=dict(size=10, color='red'))
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()


def graph_from_coord(coord: Coordinate, distance: int=500, network_type='drive') -> nx.MultiDiGraph:
    """
    Create a graph from a coordinate

    :param coord: coordinate (lon, lat)
    :param distance: distance around the coordinate
    :param network_type: network type
    :return: graph, nodes, edges
    """
    return ox.graph_from_point((coord[1], coord[0]), dist=distance, network_type=network_type)


def get_surrounding_streets(G: nx.MultiDiGraph, coordinate: Coordinate, street_name_exclusions: list[str]) -> tuple[UUIDv4, Coordinates, Indexes] | None:
    """
    Finds a closed path around a coordinate folowing the streets (city block that contains the coordinate)  
    The path is the one with the shortest number of nodes (shortest path not garantied)

    :param G: graph of the streets
    :param coordinate: initial coordinate (lon, lat) that is inside the block
    :param street_name_exclusions: list of words included in the street names to exclude from the path ("Ruelle" is excluded by default)
    :return: UUID string, list of coordinates, list of node indexes
        the UUID is based on the hash of the path and is unique for each path (no matter the start and end point)
    """
    street_name_exclusions = street_name_exclusions or []
    street_name_exclusions.append("Ruelle")

    last_edge = __search_path_algorithm(G, coordinate, street_name_exclusions, verbose=0, max_depth=10)
    if last_edge is None: return None

    path_edges = last_edge.get_path()
    path_coords, indexes = __path_to_coords_from_unordered(path_edges, simplified=False)
    uuid_str = __uuid(indexes)

    return uuid_str, path_coords, indexes


def save_surrounding_streets(uuid_str: UUIDv4, coords: Coordinates, indexes: Indexes, folder: str = None) -> None:
    """
    Saves to csv and shp the coordinates of the surrounding streets of a given coordinate.  
    Will create these files :
    - `folder/uuid.csv` : list of coordinates (lon, lat) of the streets with metadata (uuid, epsg, indexes)
    - `folder/uuid.shp` : shapefile of the streets
    - `folder/uuid.dbf` : dbf file of the streets (for the shapefile)
    - `folder/uuid.prj` : prj file of the streets (for the shapefile)
    - `folder/uuid.shx` : shx file of the streets (for the shapefile)

    :param uuid_str: UUID string
    :param coords: list of coordinates [(lon, lat), ...]
    :param indexes: list of node indexes in the path [u, v, w, ...]
    :param folder: folder to save the files in
    """
    folder = folder[:-1] if folder[-1] == '/' else folder
    path = f'{folder}/{uuid_str}' if folder else f'{uuid_str}'
    save_csv(f'{path}.csv', coords, epsg=CRS_GPS, metadata={'uuid': uuid_str, 'indexes': str(indexes).replace(',', '')})
    save_from_csv(f'{path}.csv', f'{path}.shp')


def get_sample_points(shape: tuple[int], sample_max=3, remove_half=True, force_add_center=True) -> Points:
    """
    Get a grid of sample points inside an array

    :param array: numpy array
    :param sample_max: number of points on the longest side
    :param remove_half: remove every other point (even with 0 indexing)
    :param force_add_center: add the center point if not already in the list
    :return: list of points [(y, x), ...]
    """
    sample_max = max(1, sample_max)
    height, width = shape[:2]
    long_side = max(height, width)
    short_side = min(height, width)
    ratio = long_side / short_side
    sample_small = max(1, int(math.ceil(sample_max / ratio)))

    h_sample = sample_max if height > width else sample_small
    w_sample = sample_max if width > height else sample_small
    remove_half = remove_half and (h_sample > 2 and w_sample > 2)

    h = height / (h_sample + 1)
    w = width / (w_sample + 1)

    points = []
    for i, j in [(i, j) for i in range(h_sample) for j in range(w_sample)]:
        points.append( (int(h * i + h), int(w * j + w)) )

    if remove_half: 
        points = [(0,0)] + points
        points = points[::2][1:]

    if force_add_center:
        center = (int(height / 2), int(width / 2))
        points.append(center) if center not in points else None

    return points


########## EDGE FUNCTIONS (PATH FINDING) ##########

@dataclass
class Edge:
    """
    OSM Edge data used for path finding
    """
    edge_dict: dict
    parent: 'Edge'
    children: list['Edge']

    def __eq__(self, other: 'Edge') -> bool:
        """
        :param other: other edge
        :return: True if edges are equal
        """
        return ( self.index() == other.index() ) or ( self.index() == other.index()[::-1] )

    def __contains__(self, edge: 'Edge') -> bool:
        """
        :param edge: edge to check
        :return: True if edge is in the path
        """
        return edge in self.get_path()

    def __len__(self) -> int:
        """
        :return: length of the path
        """
        return len(self.get_path())

    def add_child(self, edge: 'Edge') -> bool:
        """
        Add a child edge and set its parent

        :param edge: child edge
        :return: True if edge was added
        """
        if edge not in self.children:
            self.children.append(edge)
            edge.parent = self
            return True
        return False

    def get_path(self) -> list['Edge']:
        """
        :return: path from the current edge to the seed edge
        """
        edge = self
        path: list['Edge'] = []
        while edge is not None:
            path.append(edge)
            edge = edge.parent
        return path

    def index(self) -> tuple:
        """
        :return: start and end index (u, v)
        """
        return self.edge_dict['u'], self.edge_dict['v']

    def get_indexes(self) -> list[tuple]:
        """
        :return: list of node indexes pairs in the path [(u_0, v_0), (u_1, v_1), ...]  
            such as the v_0 does not necessarily equal u_1
        """
        return [edge.index() for edge in self.get_path()]
    
    def get_coords(self) -> tuple[list[float], list[float]]:
        """
        :return: list of coordinates ([lon, ...], [lat, ...])
        """
        x = self.edge_dict['geometry'].coords.xy[0].tolist()
        y = self.edge_dict['geometry'].coords.xy[1].tolist()
        return x, y


def __is_path_closed(edges: list[Edge]) -> bool:
    """
    :param edges: list of edges
    :return: True if the path is closed (the first and last coordinates are the same)
    """
    path_coords, _ = __path_to_coords_from_ordered(edges)
    return path_coords[0] == path_coords[-1]


def __edge_factory(u: Index, v: Index, edge_dict: dict) -> Edge:
    """
    Convert edge dict to Edge object for the path finding algorithm

    :param u: start node index
    :param v: end node index
    :param edge_dict: edge data
    :return: Edge object
    """
    edge_dict['u'] = u
    edge_dict['v'] = v
    osmid = edge_dict['osmid']
    edge_dict['osmid'] = osmid[0] if isinstance(osmid, list) else osmid
    return Edge(edge_dict, None, [])


def __get_edges_at_coordinate(G: nx.MultiDiGraph, coord: Coordinate, tolerance = 0.0) -> gpd.GeoDataFrame:
    """
    Get edges that start or end at a given coordinate  
    Bruteforce method to find the edges at a given coordinate

    :param G: graph
    :param coord: coordinate (lon, lat) EPSG:4326
    :param tolerance: tolerance in meters for floating-point precision issues (default: 0.0)
    :return: list of tuples (u, v, edge_dict)
    """
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    target_point = Point(coord)

    def matches_start_or_end(row):
        if row.geometry and row.geometry.geom_type == "LineString":
            start_point = Point(row.geometry.coords[0])
            end_point = Point(row.geometry.coords[-1])
            return start_point.distance(target_point) <= tolerance or end_point.distance(target_point) <= tolerance
        return False
    
    edges_gpd : gpd.GeoDataFrame = edges[edges.apply(matches_start_or_end, axis=1)]
    edge_list = [(i[0], i[1], row.to_dict()) for i, row in edges_gpd.iterrows()]
    return edge_list


def __str_contains(string: str, substrings: list[str] = None) -> bool:
    """
    Check if a string contains any of the substrings

    :param string: string to check
    :param substrings: list of substrings
    :return: True if string contains any of the substrings
    """
    substrings = substrings or []
    return any(substring in string for substring in substrings)


def __get_connected_edges(G: nx.MultiDiGraph, parent_edge: Edge, name_exclusions: list[str] = None, mode: str='uv') -> list[Edge]:
    """
    Connected edges (in and out) to the edge (v)

    :param G: graph
    :param parent_edge: input edge
    :param name_exclusions: list of strings to exclude if inside the child edge name (such as "Ruelle")
    :param mode: 'u' for `in` edges, 'v' for `out` edges, 'uv' for both
    :return: list of connected edges
    """
    edges = []
    if 'v' in mode:
        v = parent_edge.edge_dict['v']
        edges += list(G.in_edges(v, data=True))
        edges += list(G.out_edges(v, data=True))

    if 'u' in mode:
        u = parent_edge.edge_dict['u']
        edges += list(G.in_edges(u, data=True))
        edges += list(G.out_edges(u, data=True))
    
    for i, edge_tuple in enumerate(edges):
        u, v, edge_dict = edge_tuple
        edge = __edge_factory(u, v, edge_dict)
        if ( 'geometry' not in edge_dict ) or ('name' in edge_dict and __str_contains(edge_dict['name'], name_exclusions) ):
            edges[i] = None
            continue
        edges[i] = edge if parent_edge.add_child(edge) else None

    return [edge for edge in edges if edge is not None]


def __get_closest_edge(G: nx.MultiDiGraph, coord: Coordinate, name_exclusions: list[str] = None, distance_crs=4326) -> Edge:
    """
    Finds the closest edge to a point

    :param G: graph
    :param coord: coordinate (lon, lat) EPSG:4326
    :param name_exclusions: list of strings to exclude from edge names (such as "Ruelle")
    :param distance_crs: CRS for distance computation (default: EPSG:4326)
    :return: edge data
    """
    import warnings

    if name_exclusions is None or len(name_exclusions) == 0:
        u, v, _ = ox.distance.nearest_edges(G, X=coord[0], Y=coord[1])
    else:
        edges = ox.convert.graph_to_gdfs(G, nodes=False, edges=True)
        filtered_edges = edges[~edges["name"].astype(str).str.contains("|".join(name_exclusions), na=False, case=False)].copy()

        point_geom = gpd.GeoDataFrame(geometry=[Point(coord)], crs=CRS_GPS)
        filtered_edges = filtered_edges.to_crs(distance_crs)
        point_geom = point_geom.to_crs(distance_crs)

        # Mutes the warning in case of using EPSG:4326 for distance computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered_edges["distance"] = filtered_edges.geometry.distance(point_geom.geometry.iloc[0])

        u, v, _ = filtered_edges.loc[filtered_edges["distance"].idxmin()].name

    edge_dict = G.get_edge_data(u, v)[0]
    return __edge_factory(u, v, edge_dict)


def __stop_condition(seed_edge: Edge, edge: Edge, origin: Coordinate) -> bool:
    """
    Stop condition for the path finding

    :param seed_edge: seed edge (first edge of the path)
    :param edge: current edge
    :param origin: origin coordinate for the zone (lon, lat) EPSG:4326
    :return: True if the path is closed around the origin coordinate
    """
    seed_u = seed_edge.edge_dict['u']
    u = edge.edge_dict['u']
    v = edge.edge_dict['v']
    if v == seed_u or u == seed_u:
        path, _ = __path_to_coords_from_ordered(edge.get_path())
        return __is_path_closed(edge.get_path()) and is_inside(path, [origin])
    return False


def __path_to_coords_from_ordered(edges: list[Edge], simplified=False) -> tuple[Coordinates, Indexes]:
    """
    Converts a list of edges to an ordered list of coordinates  
    Some edges may be in reverse direction because on one-way streets  
    Coordinates are ordered to create a continuous closed path  

    uses intersection indexes to determine the direction of the path  
    verifies if the edges contain a geometry  

    :param edges: list of edges
    :param simplified: simplified path (only start and end coordinates of each edge)
    :param return_indexes: if True, return the index of the intersections of the edges in the path
    :return: list of coordinates [(lon, lat), ...] and list of indexes [u, v, ...]
    """
    lons, lats = [], []
    x, y = [], []
    indexes = []

    u, v = edges[0].index()

    for i in range(len(edges)):
        next_index = (i + 1) % len(edges)
        u_next, v_next = edges[next_index].index()

        if 'geometry' in edges[i].edge_dict:
            x, y = edges[i].get_coords()
            if simplified: x, y = [x[0], x[-1]], [y[0], y[-1]]
        else: 
            lons = np.append(lons, 0)
            lats = np.append(lats, 0)
            x, y = [], []

        undex_append = u
        if u == u_next or u == v_next:
            undex_append = v
            x.reverse()
            y.reverse()

        indexes.append(undex_append)
        lons = np.append(lons[:-1], x)
        lats = np.append(lats[:-1], y)

        u, v = u_next, v_next

    return list(zip(lons, lats)), indexes


def __pop_edge(edges: list[Edge], index_search) -> tuple[list[Edge], Edge]:
    """
    finds and pops the edge with the given intersection index
    
    :param edges: list of edges
    :param index_search: intersection index to search in the list of edges
    :return: a tuple of (edges, found_edge) or (edges, None)
    """
    for i, edge in enumerate(edges):
        u, v = edge.index()
        if u == index_search or v == index_search:
            edges.pop(i)
            return edges, edge
    return edges, None


def __path_to_coords_from_unordered(edges: list[Edge], simplified=False) -> tuple[Coordinates, Indexes]:
    """
    Converts a list of edges to an ordered list of coordinates  
    Some edges may be in reverse direction because on one-way streets  
    Coordinates are ordered to create a continuous closed path  

    uses intersection indexes to determine the direction of the path  
    verifies if the edges contain a geometry  
    reorders the edges to close the path  
    `warning` force closes the path if the path is not closed  

    :param edges: list of edges
    :param simplified: simplified path (only start and end coordinates of each edge)
    :param return_indexes: if True, return the index of the intersections of the edges in the path
    :return: list of coordinates [(lon, lat), ...] and list of indexes [u, v, ...]
    """
    edges = edges.copy()
    lons, lats = [], []
    indexes = []

    edge = edges.pop(0)
    u, v_search = edge.index()
    indexes.append(u)

    x, y = edge.get_coords()
    if simplified: x, y = [x[0], x[-1]], [y[0], y[-1]]
    lons = np.append(lons, x)
    lats = np.append(lats, y)

    while len(edges) > 0:
        indexes.append(v_search)
        edges, edge = __pop_edge(edges, v_search)
        if edge is None: break

        u, v = edge.index()
        if 'geometry' in edge.edge_dict:
            x, y = edge.get_coords()
            if simplified: x, y = [x[0], x[-1]], [y[0], y[-1]]
        else:
            lons = np.append(lons, 0)
            lats = np.append(lats, 0)
            x, y = [], []

        if v_search == v:
            x.reverse()
            y.reverse()

        lons = np.append(lons[:-1], x)
        lats = np.append(lats[:-1], y)
        v_search = u if v == v_search else v

    # close the path if not already closed
    if lons[0] != lons[-1] or lats[0] != lats[-1]:
        lons = np.append(lons, lons[0])
        lats = np.append(lats, lats[0])

    return list(zip(lons, lats)), indexes


def __get_ordered_path_indexes(edges: list[Edge]) -> Indexes:
    """
    works only if the edges are ordered and the path is closed

    :param edges: list of edges
    :return: list of ordered node indexes in the path [u, v, w, ...]
    """
    indexes = []
    u, v = edges[0].index()

    for i in range(len(edges)):
        next_index = (i + 1) % len(edges)
        u_next, v_next = edges[next_index].index()
        if u == u_next or u == v_next: indexes.append(v)
        else: indexes.append(u)
        u, v = u_next, v_next

    return indexes


def __get_ordered_path_indexes_from_unordered(edges: list[Edge]) -> Indexes:
    """
    works if the edges are not ordered and if the path is closed  
    more robust than __get_ordered_path_indexes

    :param edges: list of edges
    :return: list of node indexes in the path [u, v, w, ...]
    """
    edges = edges.copy()
    indexes = []

    edge = edges.pop(0)
    u, v_search = edge.index()
    indexes.append(u)

    while len(edges) > 0:
        indexes.append(v_search)
        edges, edge = __pop_edge(edges, v_search)
        if edge is None: break
        u, v = edge.index()
        v_search = u if v == v_search else v

    return indexes


def __uuid(indexes: Indexes) -> UUIDv4:
    """
    Generate a UUID based on the hash of the path  
    Two paths with the same edges will always have the same UUID (no matter the start and end point)

    :param indexes: list of node indexes
    :return: UUID string
    """
    data = str(sorted(indexes))
    hash_bytes = hashlib.sha256(data.encode()).digest()
    uuid_bytes = bytearray(hash_bytes[:16])

    # Set UUIDv4 version (4) and variant bits (RFC 4122 compliant)
    uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x40  # Set version to 4
    uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80  # Set variant to RFC 4122

    generated_uuid = uuid.UUID(bytes=bytes(uuid_bytes))
    return str(generated_uuid)


def __search_path_algorithm(G: nx.MultiDiGraph, seed_coord: Coordinate, name_exclusions: list[str] = None, max_depth: int=10, verbose: float=0) -> Edge | None:
    """
    Path finding algorithm to find the loop around a block using street edges

    :param G: graph
    :param seed_point: seed point (lon, lat) inside the block
    :param name_exclusions: list of strings to exclude from edge names (such as "Ruelle")
    :param max_depth: maximum depth to search (default: 10)
    :param verbose: plot the path at each step for the given time in seconds (default: 0 = no plot)
    :return: last edge of the path or the closest edge to the seed point if no path was found
    """
    queue: list[Edge] = []

    seed_edge = __get_closest_edge(G, seed_coord, name_exclusions)
    queue += __get_connected_edges(G, seed_edge, name_exclusions, mode='v')

    while len(queue) > 0 and len(queue[0]) < max_depth:
        pop_edge = queue.pop(0)
        if verbose: __plot_edge(pop_edge, seed_coord, plot_time=verbose)
        if __stop_condition(seed_edge, pop_edge, seed_coord):
            return pop_edge
        if not __is_path_closed(pop_edge.get_path()):
            queue += __get_connected_edges(G, pop_edge, name_exclusions, mode='uv')

    return None


def __plot_edge(edge: Edge, seed_coord: Coordinate, plot_time=0.5, simplified=False) -> None:
    """
    Plot an edge on the map

    :param edge: edge to plot
    :param color: edge color
    """
    from IPython.display import clear_output
    path_to_coords = __path_to_coords_from_unordered

    if plot_time > 0:
        time.sleep(plot_time)
        clear_output()

    path = edge.get_path()
    path_coords, indexes = path_to_coords(path, simplified)

    print(edge.get_indexes())
    print(path_coords)
    print(indexes)
    print(__uuid(indexes))

    plot_path(path_coords, seed_coord)


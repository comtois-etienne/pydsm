from dataclasses import dataclass
import hashlib
import uuid
import time

from osgeo import gdal, ogr, osr
from shapely.geometry import Point, Polygon
import plotly.express as px
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx


# SHAPEFILE FUNCTIONS

def open_shapefile(path: str):
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
    shapefile = open(shapefile_path)
    
    srs = osr.SpatialReference()
    srs.ImportFromWkt(gdal_file.GetProjection())
    
    layer = shapefile.CreateLayer("layer", srs, ogr.wkbPolygon)
    
    field = ogr.FieldDefn("value", ogr.OFTInteger)
    layer.CreateField(field)
    
    gdal.Polygonize(band, None, layer, 0, [], callback=None)
    
    shapefile = None  # close the shapefile


def save_from_coords(coords: list, epsg: int, shapefile_path: str) -> None:
    """
    Creates a shapefile from a list of coordinates.
    
    :param coords: List of coordinate tuples [(x1, y1), (x2, y2), ...] for points or [(x1, y1), (x2, y2), ..., (xn, yn)] for a polygon
    :param epsg: EPSG code for the coordinate system
    :param shapefile_path: Path where the shapefile will be saved
    """
    shapefile = open_shapefile(shapefile_path)

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


def save_from_csv(csv_path: str, shapefile_path: str, epsg: int = None) -> None:
    """
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
                k, v = line.strip()[1:].split('=')
                metadata[k] = v
    if 'epsg' not in metadata and epsg is None:
        raise ValueError("No EPSG code found in the CSV file. Add #epsg=NUMBER to the file.")
    epsg = epsg if epsg is not None else int(metadata['epsg'])
    save_from_coords(bounds.values.tolist(), epsg, shapefile_path)


def save_to_csv(csv_path: str, coordinates: list[tuple], epsg=4326):
    """
    Saves a list of coordinates to a csv file

    :param csv_path: path to the csv file
    :param coordinates: list of coordinates (x, y) or (lon, lat)
    :param epsg: crs of the coordinates
    """
    columns = ['x', 'y']
    df = pd.DataFrame(coordinates, columns=columns)
    df.to_csv(csv_path, index=False)
    # append string to firt line of file
    with open(csv_path, 'r') as original: data = original.read()
    with open(csv_path, 'w') as modified: modified.write(f'#epsg={epsg}\n' + data)


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


def reproject(array: np.ndarray | list[tuple], src_epsg: int, dst_epsg: int, round_to_millimeters=True) -> list[tuple]:
    """
    Converts a list of coordinates from one projection system to another

    :param array: List of coordinates to convert (x, y) or (lon, lat). Altitude (z) is ignored.
    :param src_epsg: EPSG code of the source projection system
    :param dst_epsg: EPSG code of the destination projection system
    :param round_to_millimeters: Round the coordinates to 3 decimal places (disabled for EPSG:4326 as destination)
    """
    array = np.array(array)[:,:2]

    if src_epsg == 4326: 
        array = array[:, [1, 0]]
        round_to_millimeters = False

    src = osr.SpatialReference()
    src.ImportFromEPSG(src_epsg)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(dst_epsg)
    transform = osr.CoordinateTransformation(src, dst)
    reprojected = np.array(transform.TransformPoints(array))

    if round_to_millimeters: reprojected = np.round(reprojected, 3)
    return reprojected[:, :2].tolist()


# PATH FUNCTIONS


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
        :return: list of node indexes pairs in the path [(u, v), ...] such as the v_0 does not necessarily equal u_1
        """
        return [edge.index() for edge in self.get_path()]
    
    def get_coords(self) -> tuple[list[float], list[float]]:
        """
        :return: list of coordinates ([lon, ...], [lat, ...])
        """
        x = self.edge_dict['geometry'].coords.xy[0].tolist()
        y = self.edge_dict['geometry'].coords.xy[1].tolist()
        return x, y
    
    def is_path_closed(self) -> bool:
        """
        :return: True if the path is closed (the first and last coordinates are the same)
        """
        path = self.get_path()
        coords = path_to_coords_from_indexes(path)
        return coords[0] == coords[-1]


def edge_factory(u, v, edge_dict: dict) -> Edge:
    """
    Convert edge to Edge object

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


def __get_edges_at_coordinate(G: nx.MultiDiGraph, coord: tuple, tolerance = 0.0) -> gpd.GeoDataFrame:
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
        edge = edge_factory(u, v, edge_dict)
        if ( 'geometry' not in edge_dict ) or ('name' in edge_dict and __str_contains(edge_dict['name'], name_exclusions) ):
            edges[i] = None
            continue
        edges[i] = edge if parent_edge.add_child(edge) else None

    return [edge for edge in edges if edge is not None]


def get_closest_edge(G: nx.MultiDiGraph, coord: tuple, name_exclusions: list[str] = None, distance_crs=4326) -> Edge:
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

        point_geom = gpd.GeoDataFrame(geometry=[Point(coord)], crs=4326)
        filtered_edges = filtered_edges.to_crs(distance_crs)
        point_geom = point_geom.to_crs(distance_crs)

        # Mutes the warning in case of using EPSG:4326 for distance computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered_edges["distance"] = filtered_edges.geometry.distance(point_geom.geometry.iloc[0])

        u, v, _ = filtered_edges.loc[filtered_edges["distance"].idxmin()].name

    edge_dict = G.get_edge_data(u, v)[0]
    return edge_factory(u, v, edge_dict)


def is_inside(coords: pd.DataFrame | list[tuple], coord: tuple) -> bool:
    """
    Returns True if the point is inside the polygon

    :param coords: coordinates pd.DataFrame(coords, columns=['lon', 'lat']) or [(lon, lat), ...]
    :param coord: coordinate (lon, lat) EPSG:4326
    :return: True if the coordinate is inside the polygon created by the coordinates
    """
    polygon = Polygon(coords)
    coord = Point(coord)
    return polygon.contains(coord)


def __stop_condition(seed_edge: Edge, edge: Edge, origin: tuple) -> bool:
    """
    Stop condition for the path finding

    :param seed_edge: seed edge (first edge of the path)
    :param edge: current edge
    :param origin: origin coordinate for the zone (lon, lat) EPSG:4326
    :return: True if the path is closed around the origin coordinate
    """
    path_to_coords = path_to_coords_from_indexes
    seed_u = seed_edge.edge_dict['u']
    u = edge.edge_dict['u']
    v = edge.edge_dict['v']
    if v == seed_u or u == seed_u:
        path = path_to_coords(edge.get_path())
        return is_inside(path, origin)
    return False


def path_to_coords_from_indexes(edges: list[Edge], simplified=False, return_indexes=False) -> list[int]:
    """
    Converts a list of edges to an ordered list of coordinates
    Some edges may be in reverse direction because on one-way streets
    Coordinates are ordered to create a continuous closed path

    uses intersection indexes to determine the direction of the path
    verifies if the edges contain a geometry

    :param edges: list of edges
    :param simplified: simplified path (only start and end coordinates of each edge)
    :param return_indexes: if True, return the index of the intersections of the edges in the path
    :return: list of coordinates [(lon, lat), ...] or [(lon, lat), ...], [u, v, ...]
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

    if return_indexes: return list(zip(lons, lats)), indexes
    else: return list(zip(lons, lats))


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


def path_to_coords_from_unordered_indexes(edges: list[Edge], simplified=False, return_indexes=False) -> list[int]:
    """
    Converts a list of edges to an ordered list of coordinates
    Some edges may be in reverse direction because on one-way streets
    Coordinates are ordered to create a continuous closed path

    uses intersection indexes to determine the direction of the path
    verifies if the edges contain a geometry
    reorders the edges to close the path
    `warning` : force closes the path if the path is not closed

    :param edges: list of edges
    :param simplified: simplified path (only start and end coordinates of each edge)
    :param return_indexes: if True, return the index of the intersections of the edges in the path
    :return: list of coordinates [(lon, lat), ...] or [(lon, lat), ...], [u, v, ...]
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

    if return_indexes: return list(zip(lons, lats)), indexes
    else: return list(zip(lons, lats))


def __get_ordered_path_indexes(edges: list[Edge]) -> list[int]:
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


def __get_ordered_path_indexes_from_unordered(edges: list[Edge]) -> list[int]:
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


def __uuid(indexes: list[int]) -> str:
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


def __search_path_algorithm(G: nx.MultiDiGraph, seed_coord: tuple, name_exclusions: list[str] = None, max_depth: int=10, verbose: float=0) -> Edge:
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

    seed_edge = get_closest_edge(G, seed_coord, name_exclusions)
    queue += __get_connected_edges(G, seed_edge, name_exclusions, mode='v')

    while len(queue) > 0 and len(queue[0]) < max_depth:
        pop_edge = queue.pop(0)
        if verbose: __plot_edge(pop_edge, seed_coord, plot_time=verbose)
        if __stop_condition(seed_edge, pop_edge, seed_coord):
            return pop_edge
        if not pop_edge.is_path_closed():
            queue += __get_connected_edges(G, pop_edge, name_exclusions, mode='uv')

    seed_edge.children = []
    return seed_edge


def plot_path(coordinates: list[tuple[float, float]], seed_coord: tuple[float, float]) -> None:
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


def __plot_edge(edge: Edge, seed_coord, plot_time=0.5, simplified=False) -> None:
    """
    Plot an edge on the map

    :param edge: edge to plot
    :param color: edge color
    """
    from IPython.display import clear_output
    path_to_coords = path_to_coords_from_unordered_indexes

    if plot_time > 0:
        time.sleep(plot_time)
        clear_output()

    path = edge.get_path()
    path_coords, indexes = path_to_coords(path, simplified, return_indexes=True)

    print(edge.get_indexes())
    print(path_coords)
    print(indexes)
    print(__uuid(indexes))

    plot_path(path_coords, seed_coord)


def __graph_from_coord(coord: tuple, distance: int=500, network_type='drive') -> nx.MultiDiGraph:
    """
    Create a graph from a coordinate

    :param coord: coordinate (lon, lat)
    :param distance: distance around the coordinate
    :param network_type: network type
    :return: graph, nodes, edges
    """
    return ox.graph_from_point((coord[1], coord[0]), dist=distance, network_type=network_type)


def get_surrounding_streets(coordinate: tuple, street_name_exclusions: list, search_distance=500) -> tuple[str, list[tuple[float, float]]]:
    """
    Finds a closed path around a coordinate folowing the streets
    The path is the one with the shortest number of nodes (shortest path not garantied)

    :param coordinate: initial coordinate (lon, lat) that is inside the block
    :param street_name_exclusions: list of words included in the street names to exclude from the path ("Ruelle" is excluded by default)
    :param search_distance: distance around the coordinate to search for the streets
    :return: UUID string, list of coordinates
        the UUID is based on the hash of the path and is unique for each path (no matter the start and end point)
    """
    street_name_exclusions = street_name_exclusions or []
    street_name_exclusions.append("Ruelle")

    G = __graph_from_coord(coordinate, search_distance)
    last_edge = __search_path_algorithm(G, coordinate, street_name_exclusions, verbose=0, max_depth=10)
    path_edges = last_edge.get_path()
    path_coords, indexes = path_to_coords_from_unordered_indexes(path_edges, return_indexes=True, simplified=False)
    uuid_str = __uuid(indexes)

    return uuid_str, path_coords


def save_surrounding_streets(coordinate, folder: str = None, street_name_exclusions: list=None, search_distance=500):
    """
    Saves to csv the coordinates of the surrounding streets of a given coordinate

    :param coordinate: the coordinate (lon, lat) epsg:4326
    :param folder: the folder where to save the csv file (will be named automatically)
    :param street_name_exclusions: list of words to exclude from the street names
    :param search_distance: the distance in meters to search for streets around the coordinate
    :return: the path to the saved csv file
    """
    uuid_str, coords = get_surrounding_streets(coordinate, street_name_exclusions, search_distance)
    folder = folder[:-1] if folder[-1] == '/' else folder
    path = f'{folder}/{uuid_str}' if folder else f'{uuid_str}'
    save_to_csv(f'{path}.csv', coords, epsg=4326)
    save_from_csv(f'{path}.csv', f'{path}.shp')
    return f'{path}.csv', f'{path}.shp'


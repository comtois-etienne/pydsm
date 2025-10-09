import numpy as np
import osgeo

from skimage import measure
from skimage.measure import label
from skimage.measure import regionprops
from skimage.transform import rotate
from skimage.morphology import convex_hull_image
from skimage.draw import polygon

from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter

from .obj import WavefrontObject
from .obj import WavefrontGroup
from .obj import WavefrontVertex

from .geo import to_ndarray as geo_to_ndarray
from .geo import get_scales as geo_get_scales
from .geo import get_coordinates_at_pixels as geo_get_coordinates_at_pixels
from .geo import get_coordinate_at_pixel as geo_get_coordinate_at_pixel

from .nda import shrink_mask as nda_shrink_mask
from .nda import normalize as nda_normalize
from .nda import are_overlapping as nda_are_overlapping

from .utils import *


#todo merge with cpa.get_tight_crop_values
def crop_bounding(mask: np.ndarray, ndsm: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Crop the arrays to the bounding box of the non-zero pixels in mask.

    :param mask: 2D numpy array containing a single instance mask
    :param ndsm: 2D numpy array of the nDSM
    :return: Cropped arrays and the coordinates of the bounding box (mask, ndsm, top-left corner (y, x))
    """
    y_max = np.where(mask)[0].max()
    y_min = np.where(mask)[0].min()
    x_max = np.where(mask)[1].max()
    x_min = np.where(mask)[1].min()
    return mask[y_min:y_max, x_min:x_max], ndsm[y_min:y_max, x_min:x_max], (y_min, x_min)


def get_instance_mask(instance_mask: np.ndarray, ndsm: np.ndarray, padding: int=300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the mask for a specific integer id.

    :param instance_mask: 2D numpy array of the instance mask (same size as the nDSM)
    :param ndsm: 2D numpy array of the NDSM (same size as the instance mask)
    :param padding: Padding to add to every side of the mask
    :return: 2D numpy array of the cropped mask with enough padding for rotation
    """
    instance_mask = label(instance_mask)
    ndsm = ndsm * instance_mask

    instance_mask, ndsm, origin = crop_bounding(instance_mask, ndsm)
    origin = np.array((origin[0] - padding, origin[1] - padding, 0))
    instance_mask = np.pad(instance_mask, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    ndsm = np.pad(ndsm, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    return instance_mask, ndsm, origin


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    """
    Smooth the mask using a Gaussian filter.
    :param mask: 2D numpy array containing a single instance mask
    :return: Smoothed mask
    """
    from skimage import filters
    binary = mask.astype(float)
    smoothed = filters.gaussian(binary, sigma=25)
    thresholded = smoothed > 0.5
    return thresholded


def get_contour(mask: np.ndarray, edges: int | None = 8, convex_hull=True, smooth=False, drop_last=True) -> np.ndarray:
    """
    Get the contour of a binary mask.

    :param mask: 2D numpy array
    :param edges: Number of edges to approximate the contour
    :param convex_hull: If True, the convex hull of the mask is used to get the contour
    :param smooth: If True, the mask is smoothed with a Gaussian filter
    :param drop_last: If True, the last point of the contour is dropped as it is the same as the first point
    :return: 2D numpy array of the contour as a list of points [ [y_1, x_1], ..., [y_n, x_n] ]
    """
    tolerance = 0.5

    mask = smooth_mask(mask) if smooth else mask
    chull = convex_hull_image(mask) if convex_hull else label(mask)
    contour = max(measure.find_contours(chull), key=len)
    simplified = measure.approximate_polygon(contour, tolerance)

    if edges is not None:
        while len(simplified) > (edges + 1):
            tolerance += 0.5
            simplified = measure.approximate_polygon(contour, tolerance)
    
        while len(simplified) < (edges + 1):
            tolerance -= 0.1
            simplified = measure.approximate_polygon(contour, tolerance)

    simplified = np.round(simplified, 0).astype(int)
    return simplified[:-1] if drop_last else simplified


def normalize_orientation(mask: np.ndarray, lndsm: np.ndarray, from_convex_hull=True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rotates the mask and NDSM to align the longest axis of the mask with the y-axis.

    :param mask: 2D numpy array containing a single instance mask
    :param lndsm: 2D numpy array containing a local nDSM
    :param from_convex_hull: If True, the convex hull of the mask is used to get the orientation
    :return: rotated mask, rotated nDSM, angle of rotation (in degrees)
    """
    props = regionprops(convex_hull_image(mask).astype(int) if from_convex_hull else mask)[0]
    angle = -np.degrees(props.orientation)
    lndsm_rotated = rotate(lndsm, angle, resize=False, preserve_range=True)
    mask_rotated = rotate(mask, angle, resize=False, preserve_range=True)
    return mask_rotated, lndsm_rotated, angle


def get_centerlines(lndsm: np.ndarray, mask_z: np.ndarray, scale: Scale=0.02, smooth=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the centerlines (y and x cross sections) of the local ndsm using a Gaussian filter.  

    :param lndsm: 2D numpy array containing a local nDSM
    :param mask_z: 2D numpy array containing a binary mask of the local nDSM
    :param scale: Scale of the nDSM in meters per pixel (default is 0.02 m per pixel)
    :param smooth: If True, the nDSM is smoothed with a Gaussian filter
    :return: 2D numpy array of the centerlines as a list of points (ordered by y, ordered by x)
    """
    lndsm_y = gaussian_filter(lndsm, sigma=(10, 0)) * mask_z if smooth else lndsm
    lndsm_x = gaussian_filter(lndsm, sigma=(0, 10)) * mask_z if smooth else lndsm

    y, x = np.array(lndsm.shape) // 2
    centerline_y = (lndsm_y[:, y] / scale).astype(int)
    centerline_x = (lndsm_x[x, :] / scale).astype(int)
    return centerline_y, centerline_x


def centerline_to_array(centerline: np.ndarray, padding=300, flip_y=True) -> np.ndarray:
    """
    Convert the centerline to a binary mask.

    :param centerline: 2D numpy array of the centerline. see `get_centerlines`.  
    :param padding: Padding to add to every side of the mask
    :return: 2D numpy array of the centerline as a binary mask (cross section)
    """
    width = len(centerline)
    height = np.max(centerline)
    mask = np.zeros((height, width))

    for i in range(width):
        if centerline[i] <= 0:
            continue
        if flip_y: 
            mask[-centerline[i]:, i] = 1
        else: 
            mask[:centerline[i], i] = 1

    mask = np.pad(mask, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    return mask


def __sort_contour(contour: np.ndarray) -> np.ndarray:
    """
    Sort the contour on Y and on X.

    :param contour: 2D numpy array of the contour
    :return: Y sorted contour, X sorted contour (coordinates are preserved)
    """
    sorted_on_y = np.array(sorted(contour.tolist()))
    sorted_on_x = np.array(sorted(contour[:, ::-1].tolist()))[:, ::-1]
    return sorted_on_y, sorted_on_x


def __get_index(point: np.ndarray, contour: np.ndarray) -> int:
    """
    Get the index of a point in the contour.

    :param point: 2D numpy array of the point
    :param contour: 2D numpy array of the contour
    :return: Index of the point in the contour
    """
    return np.where((contour[:, 0] == point[0]) & (contour[:, 1] == point[1]))[0][0]


def __get_next_side(side: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Get the next side of the contour.

    :param side: 2D numpy array of the side
    :param contour: 2D numpy array of the contour
    :return: 2D numpy array of the next side
    """
    length = len(contour)
    index = sorted([__get_index(side[0], contour), __get_index(side[1], contour)])
    index = index[::-1] if index[0] == 0 else index
    return np.array((contour[(index[1]+1) % length], contour[(index[1]+2) % length]))


def __get_prev_side(side: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Get the previous side of the contour.

    :param side: 2D numpy array of the side
    :param contour: 2D numpy array of the contour
    :return: 2D numpy array of the previous side
    """
    length = len(contour)
    index = sorted([__get_index(side[0], contour), __get_index(side[1], contour)])
    index = index[::-1] if index[0] == 0 else index
    return np.array((contour[(index[0]-2) % length], contour[(index[0]-1) % length]))


def __get_bottom_side(contour: np.ndarray) -> np.ndarray:
    """
    Get the bottom side (segment of 2 points) of the contour (sorted on Y).

    :param contour: 2D numpy array, list of points
    :return: 2D numpy array, 2 points of the bottom side
    """
    cz_sorted_on_y, _ = __sort_contour(contour)
    return cz_sorted_on_y[-2:]


def __get_top_side(contour: np.ndarray) -> np.ndarray:
    """
    Get the top side (segment of 2 points) of the contour (sorted on Y).
    
    :param contour: 2D numpy array, list of points
    :return: 2D numpy array, 2 points of the top side
    """
    cz_sorted_on_y, _ = __sort_contour(contour)
    return cz_sorted_on_y[:2]


def __get_top_by_area(lndsm: np.ndarray, mask: np.ndarray, area_ratio=0.2) -> np.ndarray:
    """
    Get the highest mask covering `area_ratio` of the area.

    :param lndsm: 2D numpy array containing a local nDSM
    :param mask: 2D numpy array containing the mask
    :param area_ratio: Ratio of the area to consider as the top
    :return: 2D numpy array, binary mask of the top
    """
    area = 0.0
    threshold = 0.99
    mask_area = np.sum(mask.astype(bool))
    normalized_lndsm = nda_normalize(lndsm)

    while area < area_ratio:
        top = normalized_lndsm > threshold
        area = np.sum(top) / mask_area
        threshold -= 0.01

    return top


def __get_top_by_height(lndsm: np.ndarray, height=0.8) -> np.ndarray:
    """
    Get the mask of the tree where the height is greater than the given height percentile.

    :param lndsm: 2D numpy array containing a local nDSM
    :param height: float, height percentile to consider as the top (default is 0.8)
    :return: 2D numpy array, binary mask of the top
    """
    normalized_lndsm = nda_normalize(lndsm)
    top = normalized_lndsm > height
    return top


def __get_top(lndsm: np.ndarray, mask: np.ndarray, area_ratio=0.15) -> np.ndarray:
    """
    Get the smallest top of 2 methods:
    - the highest mask covering `area_ratio` of the area. see `get_top_by_ratio`
    - the mask covering the top `1 - area_ratio` of the height. see `get_top_by_height`

    :param lndsm: 2D numpy array containing a local nDSM
    :param mask: 2D numpy array containing the mask
    :param area_ratio: Ratio of the area to consider as the top
    :return: 2D numpy array mask of the top
    """
    mask_shrinked = nda_shrink_mask(mask, shrink_factor=0.2)
    lndsm = lndsm * mask_shrinked

    top_by_ratio = __get_top_by_area(lndsm, mask_shrinked, area_ratio=area_ratio)
    top_by_height = __get_top_by_height(lndsm, height=(1 - area_ratio))
    top = np.logical_and(top_by_ratio, top_by_height)
    return top


def __get_ring_one_height(lndsm: np.ndarray, mask_z: np.ndarray, ring_0_height: float, scale: Scale) -> float:
    """
    Get the height of the ring 1 (second from the top).
    Uses the cross sections of the local nDSM of the tree in x and y to get the average height of the ring 1.

    :param lndsm: 2D numpy array containing a local nDSM
    :param mask_z: 2D numpy array containing the mask
    :param ring_0_height: float, height of the ring 0 (top) in meters
    :param scale: Scale of the nDSM in meters per pixel
    :return: Height of the ring 1
    """
    lndsm_cut = lndsm.copy()
    lndsm_cut[lndsm_cut > ring_0_height] = ring_0_height
    centerline_y, centerline_x = get_centerlines(lndsm_cut, mask_z, smooth=True, scale=scale)

    mask_y = centerline_to_array(centerline_y, 100, flip_y=False)
    mask_x = centerline_to_array(centerline_x, 100, flip_y=False)

    height_y, _ = mask_y.shape
    height_x, _ = mask_x.shape

    contour_y = get_contour(mask_y, convex_hull=True, smooth=False, edges=6)
    contour_x = get_contour(mask_x, convex_hull=True, smooth=False, edges=6)

    top_y = __get_top_side(contour_y)
    top_x = __get_top_side(contour_x)

    point_y_a = __sort_contour(__get_next_side(top_y, contour_y))[0][0]
    point_y_b = __sort_contour(__get_prev_side(top_y, contour_y))[0][0]
    point_y_avg = np.mean([point_y_a, point_y_b], axis=0)

    point_x_a = __sort_contour(__get_next_side(top_x, contour_x))[0][0]
    point_x_b = __sort_contour(__get_prev_side(top_x, contour_x))[0][0]
    point_x_avg = np.mean([point_x_a, point_x_b], axis=0)

    top_height = ((height_y - point_y_avg[0]) + (height_x - point_x_avg[0])) / 2
    avg_height = ((height_y - top_height) + (height_x - top_height)) / 2

    return round(pixel_to_height(avg_height, padding=100, scale=scale), 3)


def __get_rings_heights(lndsm: np.ndarray, mask_z: np.ndarray, ring_0: np.ndarray, scale: Scale) -> list[int]:
    """
    Get the heights of the rings :  
    - ring 0 height: highest point of the tree  
    - ring 1 height: calculated using the cross sections of the local nDSM of the tree. see `get_ring_one_height`  
    - ring 2 height: calculated as the difference between the ring 1 and ring 0
    - ring 3 height: calculated as the difference between the ring 2 and ring 1 divided by 2

    In case of tall trees :
    - Where the difference between the ring 3 and the ground is greater than 1/4 of the ring 0 elevation  
    - The ring 3 height is set to 1/4 of the ring 0 elevation  
    - The ring 2 height is set to the ring 3 height + 1/2 of the difference between the ring 1 and ring 0  

    :param lndsm: 2D numpy array containing a local nDSM
    :param mask_z: binary 2D numpy array containing the mask of the tree
    :param ring_0: list of points of the ring 0
    :param scale: Scale of the nDSM in meters per pixel
    :return: Tuple of the heights of the rings (ring_0, ring_1, ring_2, ring_3)
    """
    mask_top = np.zeros_like(mask_z)
    rr, cc = polygon(ring_0[:, 0], ring_0[:, 1], mask_top.shape)
    mask_top[rr, cc] = 1
    mask_top = mask_top.astype(bool)

    lndsm_top = lndsm * mask_top
    lndsm_top[~mask_top] = np.nan
    ring_0_height = round(np.nanmax(lndsm_top), 3)

    ring_1_height = __get_ring_one_height(lndsm, mask_z, ring_0_height, scale=scale)
    top_third_height = (ring_0_height - ring_1_height)

    # 1 1 1/2 (221)
    ring_2_height = round(max(0, ring_1_height - top_third_height), 3)
    ring_3_height = round(max(0, ring_2_height - (top_third_height / 2)), 3)

    # for tall trees : there is 1/4 of free space under the tree
    if ring_3_height > (ring_0_height / 4):
        ring_3_height = (ring_0_height / 4)
        ring_2_height = ring_3_height + (top_third_height / 2)

    rings = np.array([ring_0_height, ring_1_height, ring_2_height, ring_3_height])
    rings = np.round((rings / scale), 0).astype(int)
    return rings.tolist()


def __get_center_squares() -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulation of the center belt of squares of the RBH.   
    The indexing is always the same for every RBH.  

    :return: tuple of 2D numpy array of the center squares (squares indexes, triangles indexes)
    """
    top_indexes = np.array(range(8)) + 4
    bot_indexes = top_indexes + 12
    square_indexes = []
    triangles_indexes = []

    for i in range(8):
        square_indexes.append([
            top_indexes[i], 
            top_indexes[(i+1)%8], 
            bot_indexes[(i+1)%8], 
            bot_indexes[i]
        ])

    for square in square_indexes:
        triangles_indexes.append([square[0], square[1], square[2]])
        triangles_indexes.append([square[0], square[2], square[3]])

    return np.array(square_indexes), np.array(triangles_indexes)


def get_RBH_mesh(ndsm_array: np.ndarray, instance_mask: np.ndarray, scale: Scale) -> tuple[np.ndarray, np.ndarray]:
    """
    The RBH is divided in 4 rings :  
    - ring 0: the top of the tree parralel to the ground  
    - ring 1: the top middle of the tree parralel to the ground  
    - ring 2: the bottom middle of the tree parralel to the ground  
    - ring 3: the bottom of the tree parallel to the ground  

    Ring 0 and 3 processing :
    - The 4-sided polygons are divided in 2 triangles.  
    - Both are identical except for the height and are divided in 2 triangles.  
    - The convex hull of the mask of the top is the result of the smallest area of the 2 following methods:
        1. It represents the highest cross section covering 15% of the area of the tree.  
        2. The cross section at the 85th percentile of the height - whichever gives the smallest mask area. 

    Ring 1 and 2 processing :
    - The contour is smoothed to remove small details.  
    - The convex hull of the mask is used to get the contour.  
    - The 8-sided polygons are divided in triangles using the Delaunay triangulation.  
    - The sides of the polygon are not parallel to each other as in the standard RBH.  
    - The Delaunay method can cause the model to become concave when the Ring 1 and 3 are placed in 3D space.  

    :param ndsm_array: 2D numpy array of the NDSM (same size as the instance mask)
    :param instance_mask: 2D numpy array of the instance mask (same size as the nDSM array)
    :param scale: Scale of the nDSM in meters per pixel (default is 0.02 m per pixel)
    :return: list of points in matricial format (y, x, z) (values in pixels from the origin of the ndsm), triangles indexes
    """
    mask_z, lndsm, origin = get_instance_mask(instance_mask, ndsm_array, 400)
    mask_z, lndsm, angle = normalize_orientation(mask_z, lndsm)

    top_mask = __get_top(lndsm, mask_z)
    ring_1 = get_contour(mask_z, convex_hull=True, smooth=False, edges=8)
    ring_0 = get_contour(top_mask, convex_hull=True, smooth=False, edges=4, drop_last=True)
    ring_0_z, ring_1_z, ring_2_z, ring_3_z = __get_rings_heights(lndsm, mask_z, ring_0, scale)

    tree_trunk_origin = np.round(np.mean(ring_0, axis=0), 0).astype(int) + origin[:2]

    _, center_triangles_indexes = __get_center_squares()
    points = np.array(ring_0.tolist() + ring_1.tolist())
    tri = Delaunay(points)
    triangles_indexes = np.array(tri.simplices.copy().tolist() + (tri.simplices.copy() + len(points)).tolist() + center_triangles_indexes.tolist())

    points_top = np.array((add_z_to_points(ring_0, ring_0_z).tolist() + add_z_to_points(ring_1, ring_1_z).tolist())) # (y, x, z)
    points_bot = np.array((add_z_to_points(ring_0, ring_3_z).tolist() + add_z_to_points(ring_1, ring_2_z).tolist())) # (y, x, z)

    all_points = np.array((points_top.tolist() + points_bot.tolist())) # (y, x, z)
    rotation_center = np.array(mask_z.shape) / 2 - 0.5 # same rotation center as skimage.transform.rotate
    all_points = rotate_points(rotation_center, all_points, -angle)
    all_points = all_points + origin

    return all_points, triangles_indexes, tree_trunk_origin


def get_mask_height(instance_mask: np.ndarray, ndsm_array: np.ndarray, shrink_factor=0.2) -> float:
    """
    Returns the height of the mask in meters.  
    The mask is shrinked to remove high values at the edges of the mask as they can be part of another tree.  

    :param instance_mask: 2D numpy array of the instance mask (same size as the nDSM array)
    :param ndsm_array: 2D numpy array of the nDSM
    :param shrink_factor: Factor to shrink the mask (default is 0.2) (to remove high values at the edges of the mask)
    :return: Height of the mask in meters rounded to 3 decimal places (millimeters)
    """
    mask_z, lndsm, _ = get_instance_mask(instance_mask, ndsm_array, 400)
    mask_shrinked = nda_shrink_mask(mask_z, shrink_factor=shrink_factor)
    lndsm = lndsm * mask_shrinked
    height = np.nanmax(lndsm)
    return round(height, 3)


def get_mask_area(mask: np.ndarray, scale: Scale) -> float:
    """
    Get the area of the mask in m².

    :param mask: 2D numpy array of the mask
    :param scale: Scale of the image in meters per pixel
    :return: Area of the mask in m² rounded to 3 decimal places
    """
    v_size = mask.shape[0] * scale
    h_size = mask.shape[1] * scale
    area = v_size * h_size

    mask_area = np.sum(mask.astype(bool))
    ratio = mask_area / (mask.shape[0] * mask.shape[1])

    return round(area * ratio, 3)


def tree_modeling(
        ndsm: osgeo.gdal.Dataset, 
        dtm: osgeo.gdal.Dataset, 
        instance_masks: np.ndarray, 
        rejection_region: np.ndarray, 
        uuid: str, 
        minimum_height: float=3.0, 
        minimum_radius: float=1.5, 
        elevation_offset: float | bool=True, 
        verbose=False
    ) -> tuple[list[WavefrontObject], dict]:
    """
    Generate the wavefront objects from the instance masks of the trees.
    Trees are modeled using the triangulated RBH polyhedron fitted to the tree.

    :param ndsm: osgeo.gdal.Dataset of the nDSM image
    :param dtm: osgeo.gdal.Dataset of the DTM image (ground elevation)
    :param instance_masks: 2D numpy array of the instance masks of the trees (same size as the nDSM)
    :param rejection_region: 2D binary numpy array of trees to reject (same size as the nDSM) (value 1 = reject)
    :param uuid: UUID of the orthophoto
    :param minimum_height: Minimum height of the allowed trees in meters (default is 3.0 m)
    :param minimum_radius: Minimum radius of the allowed trees in meters (default is 1.5 m)
    :param elevation_offset: Offset of the elevation of the tree in meters  
        - if True, the elevation of the DTM is used  
        - if False, a 0 offset is used  
        - if float, the given value is used  
    :param verbose: If True, print the processing information
    :return: Tuple of the wavefront objects and the metadata of the objects
        - wavefront_objects: list of WavefrontObject
        - objects_metadata: dict of the metadata of the objects with {uuid}_{mask_id} as key
            - mask_id: Integer id of the mask
            - origin_x: X coordinate of the origin in meters (same epsg as the nDSM)
            - origin_y: Y coordinate of the origin in meters (same epsg as the nDSM)
            - origin_z: Z coordinate of the origin in meters (ground elevation from the DTM)
            - area: Area of the mask in m²
            - crown_height: Height of the crown in meters (tree top)
            - elevation_offset: Offset of the elevation in meters used for the object
    """
    scale = geo_get_scales(ndsm)[0]
    ndsm_array = geo_to_ndarray(ndsm)
    dtm_array = geo_to_ndarray(dtm)

    unique_ids = np.unique(instance_masks)
    unique_ids = unique_ids[unique_ids != 0]

    wavefront_objects = []
    objects_metadata = {}

    for mask_id in unique_ids:
        instance_mask = (instance_masks == mask_id)
        if nda_are_overlapping(rejection_region, instance_mask):
            print(f'  {mask_id} : skipped for touching the border') if verbose else None
            continue

        area = get_mask_area(instance_mask, scale=scale)
        if area < ((2*minimum_radius)**2):
            print(f'  {mask_id} : skipped for small area {area}') if verbose else None
            continue

        height = get_mask_height(instance_mask, ndsm_array)
        if height < minimum_height:
            print(f'  {mask_id} : skipped for small height {height}') if verbose else None
            continue

        print(f'  {mask_id} : processing with area {area} and height {height}') if verbose else None

        name = f'{uuid}_{mask_id}'
        points, indexes, origin = get_RBH_mesh(ndsm_array, instance_mask, scale=scale)
        points_geo = geo_get_coordinates_at_pixels(ndsm, points, precision=3)
        origin_geo = geo_get_coordinate_at_pixel(ndsm, origin, precision=3)
        elevation = round(float(dtm_array[origin[0], origin[1]]), 3)
        elv_offset_val = elevation_offset if isinstance(elevation_offset, float) else (elevation if elevation_offset else 0)

        objects_metadata[name] = {
            'mask_id': int(mask_id),
            'origin_x': origin_geo[0],
            'origin_y': origin_geo[1],
            'origin_z': elevation,
            'area': area,
            'crown_height': height,
            'elevation_offset': elv_offset_val,
        }

        group = WavefrontGroup.from_indexes_and_vertices('lod_cfd', indexes, points_geo)
        group = group.translate(WavefrontVertex(x=0, y=0, z=elv_offset_val))
        wavefront_object = WavefrontObject(name, sub_name='SolitaryVegetationObject', groups=[group])
        wavefront_objects.append(wavefront_object)

    return wavefront_objects, objects_metadata


import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from osgeo import gdal, ogr, osr
import pandas as pd
import cv2 # comment cv2/typing/__init__.py:168
from cv2 import resize as cv2_resize
from cv2 import INTER_CUBIC
from typing import Any, Optional, Tuple
import skimage.transform
from skimage.morphology import disk, binary_dilation, binary_closing, binary_opening, skeletonize
from skimage.transform import rotate as sk_rotate
from skimage.measure import label

from skimage.util import random_noise
from scipy import ndimage
from skimage import exposure


from .utils import *
import pydsm.const as const


"""
This file contains functions to manipulate numpy arrays (nda).  

In order :
- IO
- Scaling
- Normalisation
- Transformations
- Binary Masks
- Instance Segmentation Masks
- Visualisation
"""


########### IO ###########


def write_numpy(npz_path: str, *, data: Optional[Any]=None, metadata: Optional[Any]=None) -> None:
    """
    Writes data and metadata to a compressed numpy (.npz) file.  
    Data and metadata are wrapped in a numpy array before being written to the file.  
    Data and metadata are stored in the 'data' and 'metadata' keys of the .npz file.  

    :param npz_path: the path to the .npz file to write to.
    :param data: the data to write to the file. Defaults to None.
    :param metadata: the metadata to write to the file. Defaults to None.
    """
    np.savez_compressed(
        npz_path, 
        data=np.array([data]), 
        metadata=np.array([metadata]),
        types=np.array([{
            'data': type(data),
            'metadata': type(metadata),
        }])
    )


def write_numpy_napari(npz_path: str, array: np.ndarray) -> None:
    """
    Writes a numpy array to a compressed numpy (.npz) file.  
    The array is stored in the 'arr_0' key of the .npz file.

    :param npz_path: the path to the .npz file to write to.
    :param array: the numpy array to write to the file.
    """
    np.savez_compressed(npz_path, arr_0=array)


def read_numpy(npz_path: str, npz_format='pydsm') -> Tuple[Any, Any]:
    """
    Reads data and metadata from a compressed numpy (.npz) file.

    :param npz_path: str, path to the .npz file to read from.
    :param npz_format: str, format of the data. Defaults to 'pydsm'.  
        `pydsm` will return the data and metadata as they are stored in the file.  
        `napari` will return the data at `arr_0`.  
        `None` will return the data as is.  
    :return: a tuple containing the data and metadata read from the file.
    """
    npz = np.load(npz_path, allow_pickle=True)
    if npz_format == 'pydsm': 
        return npz['data'][0], npz['metadata'][0]
    if npz_format == 'napari':
        return npz['arr_0']
    return npz


def read_numpy_data(npz_path: str) -> Tuple[Any, type]:
    """
    Reads only the data from a compressed numpy (.npz) file.

    :param npz_path: str, path to the .npz file to read from.
    :return: a tuple containing the data and its type read from the file.
    """
    npz = np.load(npz_path, allow_pickle=True)
    return npz['data'][0], npz['types'][0]['data']


def read_numpy_metadata(npz_path: str) -> Tuple[Any, type]:
    """
    Reads only the metadata from a compressed numpy (.npz) file.

    :param npz_path: str, path to the .npz file to read from.
    :return: tuple of the metadata and its type read from the file.
    """
    npz = np.load(npz_path, allow_pickle=True)
    return npz['metadata'][0], npz['types'][0]['metadata']


def to_gdal(array: np.ndarray, epsg: int, origin: tuple, pixel_size: Scale = 1.0) -> gdal.Dataset:
    """
    Converts a NumPy array to a GDAL dataset.
    
    :param nda: NumPy array (2D or 3D)
    :param epsg: EPSG code of the coordinate system
    :param origin: Origin of the coordinate system (x, y) (top left corner of the array)
    :param pixel_size: Size of each pixel in meters per pixel (assumes square pixels)
    :return: GDAL dataset
    """
    # Get dimensions
    height, width = array.shape[:2]
    gdal_dtype = DTYPE_TO_GDAL[array.dtype.type]
    bands = array.shape[2] if array.ndim == 3 else 1

    # Create an in-memory GDAL dataset
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create('', width, height, bands, gdal_dtype)
    
    # Set the geotransform
    origin_x, origin_y = origin[:2]
    geotransform = (origin_x, pixel_size, 0, origin_y, 0, -pixel_size)
    dataset.SetGeoTransform(geotransform)
    
    # Set the projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dataset.SetProjection(srs.ExportToWkt())
    
    # Write array data to bands
    for i in range(bands):
        band_data = array[:, :, i] if bands > 1 else array
        dataset.GetRasterBand(i + 1).WriteArray(band_data)
    
    # Flush cache to ensure data is written
    dataset.FlushCache()
    return dataset


def save_to_wavefront(array: np.ndarray, file_path: str, origin=(0.0, 0.0), pixel_size=(1.0, 1.0)):
    """
    `Warning` works at small scale, but not at large scale  
    Converts a 2D array containing height values to a Wavefront .obj file.  

    :param array: 2D array of the dsm
    :param file_path: path to save the obj file
    :param origin: origin of the coordinate system
    :param pixel_size: size of each pixel (distance between each pixel)
    """
    # mirror the image if the pixel size is negative
    if pixel_size[0] < 0:
        array = np.flip(array, axis=1)
    if pixel_size[1] < 0:
        array = np.flip(array, axis=0)

    h, w = array.shape
    obj_lines = []

    x = np.arange(w) * abs(pixel_size[0]) + origin[0]
    y = np.arange(h) * abs(pixel_size[1]) + origin[1] # todo try without abs
    xx, yy = np.meshgrid(x, y)

    zz = array.flatten()
    vertices = np.column_stack((xx.flatten(), yy.flatten(), zz))
    obj_lines.extend(f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in vertices)

    # Create faces using indices
    for i in range(h - 1):
        for j in range(w - 1):
            # Calculate the 1D indices of the vertices
            v1 = i * w + j + 1
            v2 = v1 + 1
            v3 = v1 + w
            v4 = v3 + 1
            # Add two triangular faces for the quad
            obj_lines.append(f"f {v1} {v2} {v4}")
            obj_lines.append(f"f {v1} {v4} {v3}")

    with open(file_path, "w") as f:
        f.write("\n".join(obj_lines))


########### SCALING ###########


def downsample(array: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample the array 1 pixel is kept every factor pixels.  
    Decimation method.  
    Slower than `rescale_nearest_neighbour`. 

    :param arr: np.ndarray of shape (n, m)
    :param factor: int, factor to downsample the array
    :return: np.ndarray of shape (n // factor, m // factor)
    """
    return array[::factor, ::factor]


def upscale_nearest_neighbour(array: np.ndarray, factor: int) -> np.ndarray:
    """
    Upscale an array using nearest neighbour interpolation

    :param array: np.ndarray of shape (n, m).
    :param factor: int, factor to upscale the array.
    :return: np.ndarray of shape (n * factor, m * factor)
    """
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def rescale_nearest_neighbour(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Upscale or downscale an array using nearest neighbour interpolation.  
    About twice faster than `downsample` for the same result.  

    :param array: np.ndarray of shape (n, m).
    :param shape: tuple of the new size (y, x) (array.shape)
    :return: np.ndarray of shape (y, x)
    """
    return cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def rescale_linear(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Upscale or downscale an array using bilinear interpolation.

    :param array: np.ndarray of shape (n, m).
    :param shape: tuple of the new size (y, x) (array.shape)
    :return: np.ndarray of shape `shape`
    """
    dtype = array.dtype
    array = array.astype(np.float32)
    resized = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return resized.astype(dtype)


def crop_resize(array: np.ndarray, bbox: tuple[Coordinate, Coordinate], resolution: int) -> np.ndarray:
    """
    Crop and resize a numpy array to a given bounding box and resolution.
    
    :param array: numpy array to crop and resize
    :param bbox: bounding box as a tuple of top left (y,x) and bottom right (y,x) pixel coordinates
    :param resolution: desired resolution in pixels
    :return: cropped and resized numpy array
    """
    top_left, bottom_right = bbox
    tile = array[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    tile = cv2_resize(tile, (resolution, resolution), interpolation=INTER_CUBIC)
    return tile


########### NORMALISATION ###########


def normalize(array: np.ndarray) -> np.ndarray:
    """
    Normalize the array to [0.0, 1.0]

    :param arr: np.ndarray of shape (n, m, k).
    """
    min_val = np.min(array)
    max_val = np.max(array)
    array = array - min_val
    div = (max_val - min_val) if (max_val - min_val) != 0 else 1
    array = array / div
    return array


def clip_rescale(array: np.ndarray, clip_height: float) -> np.ndarray:
    """
    Clips out all the values above clip_height and rescales the values from `[(min_value / clip_height), 1.0]` where `1.0` equals `clip_height`

    :param array: np.ndarray dsm like array
    :param clip_height: float, value that will be mapped to 1.0 in the return matrix
    :return: np.ndarray
    """
    array = array / clip_height
    array = np.clip(array, 0, 1.0)
    return array


def to_uint8(array: np.ndarray, norm=True) -> np.ndarray:
    """
    Convert the array to an 8bit integer array.

    :param arr: np.ndarray of shape (n, m, k).
    :param norm: bool, if True, the array is normalized before conversion.
    """
    array = normalize(array) if norm else array
    return (array * 255).astype(np.uint8)


def nda_round(array: np.ndarray, decimals: int=2) -> np.ndarray:
    """
    Round the array to the specified number of decimals if the array is a float.  
    Round to the lower integer if the array is an integer.
    
    :param arr: np.ndarray of shape (n, m, k).
    :param decimals: int, number of decimals or lower int bound
    """
    assert decimals >= 0, "decimals must be a positive integer"
    if array.dtype == np.uint8:
        if decimals == 0:
            return array
        else :
            x = array // decimals
            return x.astype(int) * decimals
    else :
        x = np.round(array, decimals)
        return x.astype(int) if decimals == 0 else x


def round_to_mm(array: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Converts from meters to meters with a precision of millimeters.  
    ~ assuming the tallest building on earth is (830m) on top of mount everest (8848m)  
    ~ 9,679,000mm -> fits in 32 signed bits (to keep negative values)  
    
    :param nda: np.ndarray of shape (n, m).
    :param dtype: data type of the output array (default np.float32).
    :return: np.ndarray of shape (n, m) with values in meters.
    """
    return np.round(array, 3).astype(dtype)


def convert_2D_to_3D(array: np.array, rev=False) -> np.array:
    """
    2D array to 3D array

    :param img: 2D image
    :param rev: reverse the image so the color is applied to the black pixels
    :return: 3D image
    """
    if rev: array = 1.0 - array
    array = np.stack([array] * 3, axis=2)
    return array


########### TRANSFORMATIONS ###########


def rotate(array: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the input array by the specified angle.  
    Wrapper around `skimage.transform.rotate` to preserve dtype  
    
    :param array: Input array to be rotated.
    :param angle: Angle in degrees. If None, a random angle between 0 and 360 is chosen.
    :return: Rotated array and the angle used for rotation.
    """
    dtype = array.dtype
    array = array.astype(np.float32)
    rotated = sk_rotate(array, angle, resize=True).astype(dtype)
    return rotated


def rescale(array: np.ndarray, current_spacial_resolution: float | tuple[float, float], new_spacial_resolution: float | tuple[float, float] = 0.02) -> np.ndarray:
    """
    Sub-pixel rescaling of the array.  
    Google maps has a max pan-shapened resolution of ~15cm/px (0.15m/px)  
    Bigger value for spacial resolution means a smaller value for the new resolution.  
    
    :param arr: np.ndarray of shape (n, m).
    :param current_spacial_resolution: float, resolution of the array. (width, height) in (m/px)
    :param new_spacial_resolution: float, new resolution of the array. (width, height) in (m/px)
    :return: np.ndarray with the new resolution
    """
    if not isinstance(new_spacial_resolution, tuple):
        new_spacial_resolution = (new_spacial_resolution, new_spacial_resolution)

    if not isinstance(current_spacial_resolution, tuple):
        current_spacial_resolution = (current_spacial_resolution, current_spacial_resolution)

    w_ratio = abs(current_spacial_resolution[0] / new_spacial_resolution[0])
    h_ratio = abs(current_spacial_resolution[1] / new_spacial_resolution[1])

    # we need to keep the depth ratio (color layers) for the orthophoto
    zoom_ratio = (w_ratio, h_ratio) if array.ndim == 2 else (w_ratio, h_ratio, 1)

    return zoom(array, zoom=zoom_ratio) # todo does it also rescale the colors ???


def split_four(array: np.ndarray) -> np.ndarray:
    """
    Splits a single array into 4 identically sized tiles  

    :param array: array
    :returns: a list of 4 np.array (`top-left`, `top-right`, `bottom-left`, `bottom-right`)
    """
    h, w = array.shape[:2]
    h, w = h // 2, w // 2

    sub_arrays = []
    for i in range(2):
        for j in range(2):
            sub_arrays.append(array[i * h:(i + 1) * h, j * w:(j + 1) * w])

    return sub_arrays


def center_crop(array: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Center crop a 2D or 3D numpy array to the specified new height and width.

    :param array: np.ndarray, input array to be cropped
    :param new_height: int, desired height of the cropped array
    :param new_width: int, desired width of the cropped array
    :return: np.ndarray, center cropped array
    """
    height, width = array.shape[:2]
    start_y = (height - new_height) // 2
    start_x = (width - new_width) // 2
    return array[start_y:start_y + new_height, start_x:start_x + new_width]


def augmentation(array: np.ndarray) -> np.ndarray:
    """
    Contrast, gamma, noise and blur augmentation

    :param array: np.ndarray, rgb array to apply augmentation to
    :return: np.ndarray, augmented rgb array converted as float [0..1]
    """
    array = normalize(array.astype(float))
    contrast_adjustment = np.random.rand() > 0.25
    if contrast_adjustment:
        # print("Adjusting contrast")
        lower = np.random.uniform(0.2, 10.0)
        upper = np.random.uniform(90.0, 99.8)
        v_min, v_max = np.percentile(array, (lower, upper))
        array = exposure.rescale_intensity(array, in_range=(v_min, v_max))
        array = np.clip(array, 0.0, 1.0)

    gamma_adjustment = np.random.rand() > 0.25
    if gamma_adjustment:
        # print("Adjusting gamma")
        gamma = np.random.uniform(0.70, 0.98)
        gain = np.random.uniform(0.70, 0.98)
        array = exposure.adjust_gamma(array, gamma, gain)

    add_noise = np.random.rand() > 0.25
    if add_noise:
        # print("Adding noise")
        var = np.random.uniform(0.001, 0.01)
        array = random_noise(array, mode='gaussian', var=var)
        array = np.clip(array, 0.0, 1.0)

    add_blur = np.random.rand() > 0.25
    if add_blur:
        # print("Adding blur")
        sigma = np.random.uniform(0.5, 1.5)
        for i in range(array.shape[2]):
            l = ndimage.gaussian_filter(array[:, :, i], sigma=sigma)
            l = np.clip(l, 0.0, 1.0)
            array[:, :, i] = l

    return array


########### BINARY MASKS ###########


def get_mask(array: np.ndarray, min_mask_size = 0.02) -> np.ndarray:
    """
    Returns the mask as the smallest value of the dataset if the mask is bigger than the minimum size.

    :param nda: 2D array
    :param min_mask_size: minimum size of the mask in percentage of the total size (default 2%)
    :return: mask of the dataset if the mask is bigger than the minimum size (to avoid non-existing masks)
    """
    mask_value = np.min(array)
    if np.count_nonzero(array == mask_value) > int(array.size * min_mask_size):
        return array == mask_value
    return np.zeros_like(array, dtype=bool)


def are_overlapping(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """
    True if the two masks touch eachother by overlapping each other.

    :param mask_a: 2D binary numpy array of the first mask
    :param mask_b: 2D binary numpy array of the second mask
    :return: True if the masks touch eachother, False otherwise
    """
    return np.sum(np.logical_and(mask_a, mask_b)) > 0


def are_touching(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """
    Check if two binary masks are touching each other (side by side).

    :param mask_a: first binary mask
    :param mask_b: second binary mask
    :return: True if the masks are touching, False otherwise
    """
    dilated_a = binary_dilation(mask_a, disk(1))
    return are_overlapping(dilated_a, mask_b)


def is_mask_inside(mask_a: np.ndarray, mask_b: np.ndarray, downsample_factor=4, inside_ratio=0.8) -> bool:
    """
    Verify if any of the two mask is inside the other.  
    Inside ratio : 0.2 (spread out) -> 0.4 (clusters) -> 0.6 (dense)

    :param mask_a: np.ndarray, binary mask that can be inside mask_b
    :param mask_b: np.ndarray, binary mask that can be inside mask_a
    :param downsample_factor: bool, decimation factor to speed-up the calculation (default=4)
    :param inside_ratio: float, the overlap ratio needed to be considered as inside
    :return: True if any of the mask is inside the other, False otherwise
    """
    a = downsample(mask_a.astype(bool), downsample_factor)
    b = downsample(mask_b.astype(bool), downsample_factor)

    if np.sum(a) == 0 or np.sum(b) == 0: return False

    xor = np.logical_xor(a, b)
    aor = np.logical_or(a, b)

    # min_x, max_x, min_y, max_y = cpa.get_tight_crop_values(xor)
    # plt.imshow(xor[min_x:max_x, min_y:max_y], interpolation='nearest')
    # plt.show()

    min_area = min(np.sum(a), np.sum(b))
    diff = np.abs(np.sum(aor) - np.sum(xor))

    return diff >= (min_area * inside_ratio)


def is_mask_touching_border(mask: np.ndarray, border_width=1, top=True, bottom=True, left=True, right=True) -> bool:
    """
    Check if a binary mask is touching the borders of the array.

    :param mask: 2D numpy array (binary mask)
    :param border_width: int, width of the border to check (default: 1)
    :param top: bool, check top border (default: True)
    :param bottom: bool, check bottom border (default: True)
    :param left: bool, check left border (default: True)
    :param right: bool, check right border (default: True)
    :return: True if the mask touches any of the selected borders, False otherwise
    """
    end = (1 + border_width)
    if top and np.any(mask[0:border_width, :]):
        return True
    if bottom and np.any(mask[-end:-1, :]):
        return True
    if left and np.any(mask[:, 0:border_width]):
        return True
    if right and np.any(mask[:, -end:-1]):
        return True
    return False


def remove_holes(mask: np.ndarray) -> np.ndarray:
    """
    Remove holes of any size from the binary mask.  
    The holes are merged with the labeled region.  
    
    :param mask: np.ndarray, binay mask (containing one instance)
    :return: np.ndarray, the same binary mask with all holes removed
    """
    mask = mask.astype(int)
    labeled = label(mask, background=-1)
    mask = mask.astype(bool)

    for v in np.unique(labeled):
        hole = (labeled == v)
        if is_mask_touching_border(hole):
            continue
        hole_dilated = binary_dilation(hole, disk(1))
        if are_overlapping(mask, hole_dilated):
            mask = np.logical_or(mask, hole)

    return mask


def get_biggest_mask(masks: np.array) -> np.ndarray:
    """
    Gives the biggest mask that is not the background (0).  
    Returns empty mask if no masks are found.  

    :param masks: split binary mask, or labeled mask
    :return: binary mask of the biggest mask
    """
    masks = label(masks)
    areas = [np.sum(masks == i) for i in range(1, masks.max() + 1)]
    if len(areas) == 0:
        return np.zeros_like(masks, dtype=bool)
    biggest_mask = np.argmax(areas) + 1
    return (masks == biggest_mask)


def remove_mask_values(array: np.ndarray, min_value_ratio=0.02) -> np.ndarray:
    """
    Removes the lowest value of the array and replaces it with NaN  
    Used to display a dsm
    
    :param nda: array of the dataset
    :param min_value_ratio: minimum size of the mask in percentage of the total size (default 2%)
    :return: array of the dataset with the mask applied (mask values are set to NaN)
    """
    mask = get_mask(array, min_value_ratio)
    array[mask] = np.nan
    return array


def shrink_mask(mask: np.ndarray, shrink_factor: float = 0.1) -> np.ndarray:
    """
    Shrink the mask by a factor

    :param mask: 2D binary numpy array
    :param shrink_factor: Factor to shrink the mask (default is 0.1) (10% of the original size)
    :return: Shrinked mask
    """
    height, width = mask.shape
    height_red = int(height * shrink_factor)
    width_red = int(width * shrink_factor)

    mask_padded = np.pad(mask, ((height_red, height_red), (width_red, width_red)), mode='constant', constant_values=0)
    mask_padded = rescale_nearest_neighbour(mask_padded, (height, width))

    return mask_padded


def get_border_coords(mask: np.ndarray, decimation=512):
    """
    Get the coordinates of the border of the mask.

    :param mask: 2D binary numpy array
    :param decimation: int, decimation factor to reduce the number of points (default is 512)
    :return: np.ndarray of shape (n, 2) with the coordinates of the border of the mask.
    """
    mask = mask.astype(bool)
    mask = downsample(mask, decimation)

    mask_dilation = binary_dilation(mask, disk(1))
    result = np.logical_xor(mask, mask_dilation)

    # add exclusion border to the mask
    ones = np.ones_like(result, dtype=bool)
    ones[1:-1, 1:-1] = False
    result = np.logical_or(result, ones)

    coords = np.column_stack(np.where(result > 0))
    coords = (coords * decimation) + (decimation // 4)

    return coords


def dsm_extract_mask(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts the mask from the DSM and replaces the mask values with the minimum value of the DSM.

    :param array: np.ndarray of shape (n, m).
    :return: tuple of the array and the mask : np.ndarray of shape (n, m).
    """
    dsm_mask_val = np.min(array)
    mask = array != dsm_mask_val
    mask = mask.astype(np.float64)

    array = array.astype(np.float64)
    min_val = np.min(array[array != dsm_mask_val])
    array[array == dsm_mask_val] = min_val

    return array, mask


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    :param mask_a: np.ndarray, first binary mask
    :param mask_b: np.ndarray, second binary mask
    :return: float, IoU value
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return intersection / union


def nms(masks: list[np.ndarray], confidences: list[float] = None, iou_threshold: float = 0.5) -> list[np.ndarray]:
    """
    Perform Non-Maximum Suppression (NMS) on a list of binary masks based on their IoU.

    :param masks: list of np.ndarray, list of binary masks
    :param confidences: list of float, list of confidence scores corresponding to each mask
    :param iou_threshold: float, IoU threshold for suppression
    :return: list of np.ndarray, list of masks after NMS (from most to least confident)
    """
    if confidences is not None:
        ordered_indices = np.argsort(confidences)[::-1] # high to low
        masks = [masks[i] for i in ordered_indices]
        confidences = [confidences[i] for i in ordered_indices]

    kept_masks = []

    while len(masks) > 0:
        mask = masks.pop(0)
        kept_masks.append(mask)
        ious = [iou(mask, m) for m in masks]
        masks = [m for i, m in enumerate(masks) if ious[i] < iou_threshold]

    return kept_masks


########### INSTANCE SEGMENTATION MASKS ###########


def relabel(instances: np.ndarray) -> np.ndarray:
    """
    Relabels the input array by replacing each unique value with its index in the sorted unique values.  
    Preserve the same order of the labels.  
    This does not use the label() function from skimage.  
    Two elements with the same value will be replaced by the same index.  
    This removes gaps in the labels and ensures that the labels are contiguous.  
    Does not relabel 0 if it exists or not as it is used as a background label.  
    
    :param instances: Input numpy array with labels to be relabeled.
    :return: numpy array with relabeled values.
    """
    array = instances.copy()
    unique_values = np.unique(array)

    if unique_values[0] == 0:
        unique_values = unique_values[1:]

    for i, value in enumerate(unique_values):
        array[array == value] = (i + 1)

    return array


def get_labels_centers(instances: np.ndarray) -> list:
    """
    Get the centers of instances in a segmentation mask.  
    The center point is garanteed to be inside the instance mask.  
    Selects the median x value on the median y value. (y median values are selected first - then x on that y value)  
    The background value `0` is ignored.  
    Can be done with a downsampled version of the labels.  

    :param instances: Segmentation mask with instance labels.
    :return: List of (y, x) coordinates for each instance center.
    """
    def median(list):
        """
        `np.median` is not used as it returns the average of the two middle values when the list has an even number of elements
        """
        array = np.sort(list)
        return array[len(array) // 2]

    unique_labels = np.unique(instances)
    unique_labels = unique_labels[unique_labels != 0]
    centers = []

    for label in unique_labels:
        instance_mask = (instances == label)
        y, x = np.where(instance_mask)
        y_median = int(median(y))
        y_indexes = np.where(y == y_median)
        x_values = x[y_indexes]
        x_median = int(median(x_values))
        centers.append((y_median, x_median))

    return centers


def get_centers(instances: np.ndarray, downsampling_size=100):
    """
    Returns the centers of the instances in the labels array.

    :param instances: np.ndarray with shape (H, W) representing the instance segmentation labels.
    :param downsampling_size: int, size to which the labels will be downsampled.
    :return: tuple of two np.ndarrays:
        - array_centers: binary numpy array with the centers of the instances set to `1.0` (size of `labels`)
        - array_centers_downsampled: binary numpy array with the centers of the downsampled instances (size `downsampling_size`)
    """
    labels_downsampled = rescale_nearest_neighbour(instances, (downsampling_size, downsampling_size))
    centers = np.array(get_labels_centers(labels_downsampled))

    if centers.size == 0:
        return np.zeros_like(instances), np.zeros((downsampling_size, downsampling_size))

    array_centers_downsampled = np.zeros_like(labels_downsampled)
    array_centers_downsampled[centers[:, 0], centers[:, 1]] = 1

    centers = centers * (instances.shape[0] // downsampling_size)
    array_centers = np.zeros_like(instances)
    array_centers[centers[:, 0], centers[:, 1]] = 1

    return array_centers, array_centers_downsampled


def replace_value_inplace(instances: np.ndarray, old_values: list, new_values: list) -> None:
    """
    Replaces the specified values in the array `old_values` with their corresponding values in `new_values`

    :param array: numpy array, dtype should be integer
    :param old_values: list of values to be replaced (length must match new_values)
    :param new_values: list of new values to replace old values with (length must match old_values)
    :raises ValueError: if lengths of old_values and new_values do not match
    """
    array = instances.copy()
    old_values = np.array(old_values)
    new_values = np.array(new_values)

    if len(old_values) == 0 or len(new_values) == 0:
        return array

    max_val = max(old_values.max(), new_values.max())
    old_values += (max_val + 1)
    array += (max_val + 1)

    if len(old_values) != len(new_values):
        raise ValueError("Length of old_values and new_values must match.")
    
    for old, new in zip(old_values, new_values):
        array[array == old] = new

    array[array == (max_val + 1)] = 0
    array[array > max_val] -= (max_val + 1)

    return array


def remove_small_masks(instances: np.ndarray, min_area=const.MIN_MASK_SIZE) -> np.ndarray:
    """
    Removes all instances with an area lower or equal to `min_area`
    The instances are relabeled to find unconnected parts of instances

    :param instances: np.ndarray, array with instances to be filtered
    :param min_area: int, rejects masks with lower or equal area (default=20*20=400)
    :return: np.ndarray, array with small masks removed
    """
    labeled = label(instances)
    for v in np.unique(labeled):
        mask = (labeled == v)
        if np.sum(mask) > min_area:
            continue
        instances = instances * ~mask
    return instances


def clean_mask_instances(instances: np.ndarray, min_area=const.MIN_MASK_SIZE, remove_cracks=const.REMOVE_CRACKS_SIZE) -> np.ndarray:
    """
    Simplifies the shape of the instance masks - removes cracks and aberations  
    Masks index should be from least confident=1 to most confident=n.  
    Some masks might be removed if too small or considered as aberant (thin lines)  

    :param instances: np.ndarray, uint instance segmentation masks
    :param min_area: int, minimum area (in pixels) for keeping an instance mask. Default is 400
    :param remove_cracks: int, the crack size and small lines in the masks to be removed
    :return: np.ndarray, the instance segmentation masks without aberations
    """
    instances = remove_small_masks(instances, min_area=min_area)
    new_instances = np.zeros_like(instances)
    k = disk(remove_cracks)

    for i in np.unique(instances):
        if i == 0: continue
        mask = (instances == i)
        mask = binary_opening(mask, k)
        mask = get_biggest_mask(mask)
        mask = binary_closing(mask, k)
        mask = remove_holes(mask)
        new_instances[mask] = i

    return relabel(new_instances)


def remove_instances_below(instances: np.ndarray, depth: np.ndarray, min_depth: float) -> np.ndarray:
    """
    Remove instances in the instance segmentation mask `instances` that have a maximum depth below `min_depth` in the depth map `depth`.  

    :param instances: np.ndarray, instance segmentation mask of shape (H, W) with integer labels for each instance
    :param depth: np.ndarray, depth map of shape (H, W) with depth values in meters (ndsm)
    :param min_depth: float, minimum depth threshold in meters
    :return: np.ndarray, cleaned instance segmentation mask
    """
    cleaned_labels = instances.copy()
    for v in np.unique(instances):
        if v == 0: continue
        d = depth * (instances == v)
        if np.max(d) < min_depth:
            cleaned_labels[cleaned_labels == v] = 0
    return cleaned_labels


def stack_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """
    Combine two arrays of same size vertically.

    :param top: top instace segmentation mask
    :param bottom: bottom instance segmentation mask
    :return: relabeled combined instance segmentation mask (no merging of instances)
    """
    mask = (bottom != 0)
    bottom = bottom + np.max(top) * mask
    stacked = np.vstack((top, bottom))
    return relabel(stacked)


def stack_horizontal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Combine two arrays of same size horizontally.

    :param left: left instace segmentation mask
    :param right: right instance segmentation mask
    :return: relabeled combined instance segmentation mask (no merging of instances)
    """
    mask = (right != 0)
    right = right + np.max(left) * mask
    stacked = np.hstack((left, right))
    return relabel(stacked)


def get_border_touching_instances(instances: np.ndarray, *, tolerance=const.PIXEL_TOLERANCE, top=True, bottom=True, left=True, right=True) -> np.ndarray:
    """
    Split the instances into two masks: border-touching and non-border-touching (inside) instances.

    :param instances: instance segmentation mask
    :param top: whether to consider the top border
    :param bottom: whether to consider the bottom border
    :param left: whether to consider the left border
    :param right: whether to consider the right border
    :return: (border_instances, inside_instances)
    """
    border_instances = np.zeros_like(instances)
    inside_instances = np.zeros_like(instances)

    for v in np.unique(instances):
        if v == 0: continue
        mask = (instances == v)
        if is_mask_touching_border(mask, tolerance, top, bottom, left, right):
            border_instances += (mask * v)
        else:
            inside_instances += (mask * v)

    return border_instances, inside_instances


def get_complete_mask(mask: np.ndarray, instances: np.ndarray, pixel_tolerance=const.PIXEL_TOLERANCE, circle_tolerance=const.CIRCLE_TOLERANCE) -> np.ndarray:
    """
    1. Dilate the input mask by the given tolerance.
    2. Find which instance in `instances` touches with the dilated mask.
        - assume that the 2 masks are parts of a circle
        - compute the arc lenght of the touching part
        - compare it to the reference arc lenght
    3. Combine both masks (original masks, not dilated)
    4. Remove cracks to merge them properly.

    :param mask: binary mask to find its match in instances (shape must be the same as instances)
    :param instances: instance segmentation mask (one of which the mask might be part of) (labels are ordered as least to most confident)
    :param pixel_tolerance: tolerance in pixels to dilate the masks in order to find touching instances
    :param circle_tolerance: tolerance in deviation from perfect circle to accept a match
    :return: complete binary mask corresponding to the instance in which the input mask is found (might be the same as input mask)
    """
    def dev(v):
        return v - 1.0 if v > 1.0 else 1.0 - v
    
    unique = np.unique(instances)
    unique = unique[unique != 0]
    dilated = binary_dilation(mask, disk(pixel_tolerance))

    touching_instances = [(instances == v) for v in unique if are_touching(dilated, (instances == v))]
    touching_lenghts = [skeletonize(np.logical_and(dilated, m)).sum() for m in touching_instances]
    reference_arc = [circle_arc_lenght(np.sum(mask), np.sum(m)) for m in touching_instances]
    deviations = np.abs(np.array(touching_lenghts) / np.array(reference_arc))
    deviations = [dev(d) for d in deviations]

    best_mask = touching_instances[np.argmin(deviations)] if len(deviations) > 0 else None
    best_deviation = np.min(deviations) if len(deviations) > 0 else None

    if best_mask is not None and best_deviation <= circle_tolerance:
        mask = np.logical_or(mask, best_mask)
        mask = binary_closing(mask, disk(pixel_tolerance))
        mask = get_biggest_mask(mask)

    return mask


def _combine_instances(first: np.ndarray, second: np.ndarray, vertical_direction=True) -> np.ndarray:
    """
    1. find the border-touching instances for both instances
    2. add padding of zeroes to both masks (so they can be overlapped)
    3. for each border-touching instance in first, find its matching instance in second

    :param first: instance segmentation mask on the left or top
    :param second: instance segmentation mask on the right or bottom
    :return: combined instance segmentation mask
    """
    v = vertical_direction
    stack_func = stack_vertical if vertical_direction else stack_horizontal

    first_b, first_i = get_border_touching_instances(first, top=False, bottom=v, left=False, right=(not v))
    second_b, second_i = get_border_touching_instances(second, top=v, bottom=False, left=(not v), right=False)

    zeroes = np.zeros_like(first)
    instances = stack_func(first_i, second_i)

    first = stack_func(first_b, zeroes)
    second = stack_func(zeroes, second_b)

    for v in np.unique(first):
        if v == 0: continue
        mask = (first == v)
        complete_mask = get_complete_mask(mask, second)
        instances[complete_mask] = np.max(instances) + 1

    for v in np.unique(second):
        if v == 0: continue
        mask = (second == v)
        complete_mask = get_complete_mask(mask, first)
        instances[complete_mask] = np.max(instances) + 1

    instances = remove_small_masks(instances)
    return relabel(instances)


def combine_horizontal(instances_left: np.ndarray, instances_right: np.ndarray) -> np.ndarray:
    """
    Combine two instance segmentation masks horizontally by merging border-touching instances.  
    See `_combine_instances` function.  
    
    :param instances_left: instance segmentation mask to the left
    :param instances_right: instance segmentation mask to the right
    :return: combined instance segmentation mask in one horizontal array
    """
    return _combine_instances(
        instances_left, 
        instances_right, 
        vertical_direction=False
    )


def combine_vertical(instances_top: np.ndarray, instances_bottom: np.ndarray) -> np.ndarray:
    """
    Combine two instance segmentation masks vertically by merging border-touching instances.  
    See `_combine_instances` function.  
    
    :param instances_top: instance segmentation mask to the top
    :param instances_bottom: instance segmentation mask to the bottom
    :return: combined instance segmentation mask in one vertical array
    """
    return _combine_instances(
        instances_top, 
        instances_bottom, 
        vertical_direction=True
    )


def combine_four_instances(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Combine four instance segmentation masks into one.  
    We assume that the instances are circular to help with matching.  

    :param arrays: list of 4 instances tiles (`top-left`, `top-right`, `bottom-left`, `bottom-right`)
    :param pixel_tolerance: tolerance in pixels to dilate the masks in order to find touching instances
    :param circle_tolerance: tolerance in deviation from perfect circle to accept a match
    :return: combined instance segmentation mask (size is sum of inputs)
    """
    top = combine_horizontal(arrays[0], arrays[1])
    bot = combine_horizontal(arrays[2], arrays[3])
    full = combine_vertical(top, bot)
    return full


########### VISUALISATION ###########


def to_cmap(array: np.ndarray, cmap: str='viridis', nrm=True) -> np.ndarray:
    """
    Applies the colormap to the array.
    
    :param array: np.ndarray of shape (n, m, 1).
    :param cmap: str, name of the matplotlib colormap.
    :param nrm: bool, if True, the array is normalized.
    """
    cmap = mpl.colormaps[cmap]
    array = normalize(array) if nrm else array
    array = plt.cm.get_cmap(cmap)(array)
    return array[:, :, :3]


def dsm_to_cmap(array: np.ndarray, cmap: str='viridis') -> np.ndarray:
    """
    The 1 layer image is converted to a color image using a colormap.  
    We assume linear values in float. The smallest value is used as a mask.
    
    :param dsm: np.ndarray of shape (n, m).
    :param cmap: str, name of the matplotlib colormap.
    :return: np.ndarray of shape (n, m, 4) with the last channel as a transparency mask.
    """
    array, mask = dsm_extract_mask(array)
    array = to_cmap(array, cmap)
    array = np.dstack((array, mask))
    array = (array * 255).astype(np.uint8)
    return array


def overlay_values(array: np.ndarray, factor=64, font_size: float = 1.0, round_value=1, cmap: str = 'viridis', font_path='BAHNSCHRIFT.TTF') -> np.ndarray:
    """
    Add (display) the values of each pixel to the image

    :param arr: 2D+ array
    :param factor: nearest neighbour upscale factor
    :param font_size: font size multiplier (default 1.0 will be 1/3 of the pixel height)
    :param round_value: round the value to this number of decimal places
    :param cmap: colormap to use if the array is 2D
    :param font_path: path to the font file (default is BAHNSCHRIFT.TTF)
    """
    from PIL import Image, ImageDraw, ImageFont

    upscaled = to_uint8(array)
    upscaled = upscale_nearest_neighbour(upscaled, factor)
    if len(upscaled.shape) == 2:
        upscaled = to_cmap(upscaled, cmap)
        upscaled = to_uint8(upscaled)

    # Drawing functions
    pil_image = Image.fromarray(upscaled)
    font_size = int(font_size * factor * 0.33)
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(pil_image)

    array = nda_round(array, round_value)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            background = upscaled[i * factor, j * factor]
            color = (0,0,0) if background.mean() >= 128 else (255,255,255)
            value = [array[i, j]] if len(array.shape) == 2 else array[i, j]
            for k, e in enumerate(value):
                draw.text(((j * factor) + 1, (i * factor) + (k * font_size) + 1), str(e), fill=color, font=font)
    
    return np.array(pil_image)


def random_dtm(size: int=512, skip=1) -> np.ndarray:
    """
    Create a random DSM with a mask
    
    :param size: size of the DTM
    :param skip: skip factor for the scaling of the random DTM
    :return: random DTM
    """
    import skimage.filters as filters

    # log base 2 of the size
    log_size = int(np.log2(size))
    render_size = 2 ** log_size

    array = np.full((render_size, render_size), 0.0)
    for e in range(skip, log_size + 1):
        # print(2 ** e)
        sub_a = np.random.rand(2 ** e, 2 ** e)
        sub_a = normalize(sub_a)
        sub_a = upscale_nearest_neighbour(sub_a, 2 ** (log_size - e))
        # sub_a = filters.gaussian(sub_a, sigma=(1 + log_size - e)**2) # clouds with cmap Blues
        # sub_a = filters.gaussian(sub_a, sigma=(log_size / e))
        array = array + sub_a / e
        # array = array + sub_a / (e**2)

    return normalize(array)


class SubimageGenerator:
    """
    Splits a numpy array into patches of a given size with a specified overlap.
    The generator yields the coordinates of the top-left corner of each patch (y, x) in matricial coordinates
    and the corresponding subimage.
    The patches are generated in a row-major order, starting from the top-left corner of the array.
    The generator stops when all patches have been yielded.
    """

    def __init__(self, array: np.ndarray, size: int = 1024, min_overlap: float = 0.25):
        """
        :param array: The input numpy array to be split into patches.
        :param size: The size of the patches to be generated.
        :param min_overlap: The minimum overlap between adjacent patches, expressed as a percentage of the patch size.
            the overlap is calculated as (size - size * min_overlap)
            the actual overlap is greater or equal to this value to ensure that the patches fit within the array.
        """

        self.array = array
        self.size = size
        self.min_overlap = min_overlap

        self.height, self.width = array.shape[:2]
        self.count_width = self._get_img_count(self.width)
        self.count_height = self._get_img_count(self.height)
        self.offset_width = int((self.width - size) / (self.count_width - 1))
        self.offset_height = int((self.height - size) / (self.count_height - 1))

        self.i = 0
        self.j = 0

    def _get_img_count(self, length: int) -> int:
        return 1 + math.ceil((length - self.size) / (self.size - self.size * self.min_overlap))

    def __iter__(self):
        return self

    def __next__(self) -> tuple[tuple[int, int], np.ndarray]:
        """
        Returns the coordinates of the top-left corner of the patch (y, x) in matricial coordinates
        and the corresponding subimage.
        :return: A tuple containing the coordinates (y, x) and the subimage.
        :raises StopIteration: When all patches have been yielded.
        """
        while self.i < self.count_height:
            while self.j < self.count_width:
                y = self.i * self.offset_height
                x = self.j * self.offset_width
                self.j += 1

                subimage = self.array[y:y+self.size, x:x+self.size]
                return (y, x), subimage

            self.j = 0
            self.i += 1

        raise StopIteration


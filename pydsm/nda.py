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
from skimage.morphology import disk, binary_dilation


from .utils import *


# IO

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


# Functions

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


def clip_rescale(array: np.ndarray, clip_height: 30.0):
    """
    Clips out all the values above clip_height and rescales the values from `[(min_value / clip_height), 1.0]` where `1.0` equals `clip_height`

    :param array: np.ndarray dsm like array
    :param clip_height: float, value that will be mapped to 1.0 in the return matrix
    :return: np.ndarray
    """
    array = array / clip_height
    array = np.clip(array, 0, 1.0)
    return array


def to_uint8(array: np.ndarray) -> np.ndarray:
    """
    Convert the array to an 8bit integer array.

    :param arr: np.ndarray of shape (n, m, k).
    """
    norm = normalize(array)
    return (norm * 255).astype(np.uint8)


def relabel(labels: np.ndarray) -> np.ndarray:
    """
    Relabels the input array by replacing each unique value with its index in the sorted unique values.  
    This does not use the label() function from skimage.  
    Two elements with the same value will be replaced by the same index.  
    This removes gaps in the labels and ensures that the labels are contiguous.  
    Does not relabel 0 if it exists or not as it is used as a background label.  
    
    :param array: Input numpy array with labels to be relabeled.
    :return: numpy array with relabeled values.
    """
    array = labels.copy()
    unique_values = np.unique(array)

    if unique_values[0] == 0:
        unique_values = unique_values[1:]

    for i, value in enumerate(unique_values):
        array[array == value] = (i + 1)

    return array


def get_labels_centers(labels: np.ndarray) -> list:
    """
    Get the centers of instances in a segmentation mask.  
    The center point is garanteed to be inside the instance mask.  
    Selects the median x value on the median y value. (y median values are selected first - then x on that y value)  
    The background value `0` is ignored.  
    Can be done with a downsampled version of the labels.  

    :param labels: Segmentation mask with instance labels.
    :return: List of (y, x) coordinates for each instance center.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    centers = []

    for label in unique_labels:
        instance_mask = (labels == label)
        y, x = np.where(instance_mask)
        y_median = int(np.median(y))
        y_indexes = np.where(y == y_median)
        x_values = x[y_indexes]
        x_median = int(np.median(x_values))
        centers.append((y_median, x_median))

    return centers


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


def replace_value_inplace(array: np.ndarray, old_values: list, new_values: list) -> None:
    """
    Replaces the specified values in the array `old_values` with their corresponding values in `new_values`

    :param array: numpy array, dtype should be integer
    :param old_values: list of values to be replaced (length must match new_values)
    :param new_values: list of new values to replace old values with (length must match old_values)
    :raises ValueError: if lengths of old_values and new_values do not match
    """
    array = array.copy()
    old_values = np.array(old_values)
    new_values = np.array(new_values)
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


def are_touching(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """
    True if the two masks touch eachother.

    :param mask_a: 2D binary numpy array of the first mask
    :param mask: 2D binary numpy array of the second mask
    :return: True if the masks touch eachother, False otherwise
    """
    return np.sum(np.logical_and(mask_a, mask_b)) > 0


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

    return zoom(array, zoom=zoom_ratio)


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
    array = array.astype(np.float32)
    return cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


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


def get_border_coords(mask, decimation=512):
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


def __gaussian_blur_from_boxes(array: np.ndarray, boxes: pd.DataFrame, sigma: float = 5.0) -> np.ndarray:
    """
    :param ndarray: 3D array of the image
    :param boxes: DataFrame with the bounding boxes (xmin, ymin, xmax, ymax)
    :param sigma: sigma of the Gaussian filter
    :return: 3D array of the image with blurred bounding boxes
    """
    from skimage.filters import gaussian

    array = normalize(array)
    mask = np.zeros_like(array, dtype=np.float64)
    means = array.copy()
    for _, obj in boxes.iterrows():
        xmin, ymin, xmax, ymax = obj[['xmin', 'ymin', 'xmax', 'ymax']]
        mean = np.mean(array[ymin:ymax, xmin:xmax], axis=(0, 1))
        mask[ymin:ymax, xmin:xmax] = 1
        means[ymin:ymax, xmin:xmax] = mean

    blurred = gaussian(array, sigma=sigma)
    mask = gaussian(mask, sigma=sigma)

    new_array = array * (1 - mask) + blurred * mask
    new_array = (new_array + means) / 2

    return new_array


def anonymise_with_yolov8n(array: np.ndarray) -> np.ndarray:
    """
    Blur the bounding boxes humans in the image using a Gaussian filter.
    
    :param ndarray: 3D array of the image
    :return: 3D array of the image with blurred bounding boxes
    """
    from ultralytics import YOLO
    from pydsm.yolo import ObjectDetector

    yolo_v8 = YOLO('yolov8n.pt')
    detector = ObjectDetector(yolo_v8)
    size = (max(array.shape) + 32) // 32 * 32
    detector.detect(array, imgsz=size)
    boxes = detector.get_objs_by_name('person', 0.2)
    return __gaussian_blur_from_boxes(array, boxes)


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


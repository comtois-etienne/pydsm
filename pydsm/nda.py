import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from osgeo import gdal, ogr, osr
import pandas as pd
from typing import Any, Optional, Tuple
import skimage
import os


DTYPE_TO_GDAL = {
    int: gdal.GDT_Byte, # 1
    float: gdal.GDT_Float64, # 7
    np.int64: gdal.GDT_Int64, # 13
    np.uint8: gdal.GDT_Byte, # 1
    np.float32: gdal.GDT_Float32, # 6
    np.float64: gdal.GDT_Float64, # 7
}


# IO

def write_numpy(npz_path: str, *, data: Optional[Any]=None, metadata: Optional[Any]=None) -> None:
    """
    Writes data and metadata to a compressed numpy (.npz) file.
    Data and metadata are wrapped in a numpy array before being written to the file.
    Data and metadata are stored in the 'data' and 'metadata' keys of the .npz file.

    Parameters:
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


def read_numpy(npz_path: str) -> Tuple[Any, Any]:
    """
    Reads data and metadata from a compressed numpy (.npz) file.
    :param npz_path: str, path to the .npz file to read from.
    :return: a tuple containing the data and metadata read from the file.
    """
    npz = np.load(npz_path, allow_pickle=True)
    return npz['data'][0], npz['metadata'][0]


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


def to_uint8(array: np.ndarray) -> np.ndarray:
    """
    Convert the array to an 8bit integer array.
    :param arr: np.ndarray of shape (n, m, k).
    """
    norm = normalize(array)
    return (norm * 255).astype(np.uint8)


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


# todo: rename to nda_round
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

    return zoom(array, zoom=(w_ratio, h_ratio), order=3)


def upscale_nearest_neighbour(array: np.ndarray, factor: int) -> np.ndarray:
    """
    Upscale an array using nearest neighbour interpolation
    :param array: np.ndarray of shape (n, m).
    :param factor: int, factor to upscale the array.
    :return: np.ndarray of shape (n * factor, m * factor).
    """
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


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


def to_gdal(array: np.ndarray, epsg: int, origin: tuple, pixel_size: float = 1.0) -> gdal.Dataset:
    """
    Converts a NumPy array to a GDAL dataset.
    
    :param nda: NumPy array (2D or 3D)
    :param epsg: EPSG code of the coordinate system
    :param origin: Origin of the coordinate system (x, y) (top left corner of the array)
    :param pixel_size: Size of each pixel (assumes square pixels)
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


def save_to_wavefront(array: np.ndarray, file_path, origin=(0.0, 0.0), pixel_size=(1.0, 1.0)):
    """
    Warning : works at small scale, but not at large scale
    Converts a 2D array containing height values to a Wavefront .obj file.
    :param ndarray: 2D array of the dsm
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


def save_subimages(array: np.ndarray, output_folder: str, size: int = 2800, min_overlap: float = 0.5) -> None:
    """
    Cuts the array into overlapping subimages and saves them in the output folder.

    :param ndarray: array to be cut into overlapping subimages
    :param output_folder: folder to save the subimages
    :param size: size of the subimages
    :param min_overlap: minimum overlap between the subimages
        overlap might be bigger to keep a costant spacing between the subimages until the last one
    :return: None, saves the subimages in the output folder
    """
    def __get_img_count(length: int, size: int, min_overlap: float) -> int:
        return 1 + math.ceil( (length - size) / (size - size * min_overlap) )

    height, width = array.shape[:2]
    count_width = __get_img_count(width, size, min_overlap)
    count_height = __get_img_count(height, size, min_overlap)
    offset_width = int((width - size) / (count_width - 1))
    offset_height = int((height - size) / (count_height - 1))

    for i in range(count_height):
        for j in range(count_width):
            x = j * offset_width
            y = i * offset_height
            name = f'{y}_{x}.png'
            subimage = array[y:y+size, x:x+size]
            # will use the alpha layer if it exists
            layer_subimage = subimage if len(subimage.shape) == 2 else subimage[:, :, -1]
            layer_count = 1 if len(subimage.shape) == 2 else subimage.shape[2]
            perc_not_zero = np.count_nonzero(layer_subimage) / (subimage.size // layer_count)
            if perc_not_zero > min_overlap:
                subimage = to_uint8(subimage)
                skimage.io.imsave(os.path.join(output_folder, name), subimage)


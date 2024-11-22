import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr


def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize the array to [0.0, 1.0]
    
    :param arr: np.ndarray of shape (n, m, k).
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    arr = arr - min_val
    div = (max_val - min_val) if (max_val - min_val) != 0 else 1
    arr = arr / div
    return arr


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


def dsm_to_cmap(arr: np.ndarray, cmap: str='viridis') -> np.ndarray:
    """
    The 1 layer image is converted to a color image using a colormap.
    We assume linear values in float. The smallest value is used as a mask.
    
    :param dsm: np.ndarray of shape (n, m).
    :param cmap: str, name of the matplotlib colormap.
    :return: np.ndarray of shape (n, m, 4) with the last channel as a transparency mask.
    """
    dsm_mask_val = np.min(arr)
    mask = arr != dsm_mask_val
    mask = mask.astype(np.float64)

    arr = arr.astype(np.float64)
    min_val = np.min(arr[arr != dsm_mask_val])
    arr[arr == dsm_mask_val] = min_val

    arr = to_cmap(arr, cmap)
    arr = np.dstack((arr, mask))
    arr = (arr * 255).astype(np.uint8)
    return arr


def to_gdal(nda: np.ndarray, epsg: int, origin: tuple, pixel_size: float = 1.0) -> gdal.Dataset:
    """
    Converts a NumPy array to a GDAL dataset.
    
    :param nda: NumPy array (2D or 3D)
    :param epsg: EPSG code of the coordinate system
    :param base_coordinate: Origin of the coordinate system (x, y) (bottom left corner of the array)
    :param pixel_size: Size of each pixel (assumes square pixels)
    :return: GDAL dataset
    """
    # Get dimensions
    height, width = nda.shape[:2]
    bands = nda.shape[2] if nda.ndim == 3 else 1

    # Create an in-memory GDAL dataset
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create('', width, height, bands, gdal.GDT_Float32)
    
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
        band_data = nda[:, :, i] if bands > 1 else nda
        dataset.GetRasterBand(i + 1).WriteArray(band_data)
    
    # Flush cache to ensure data is written
    dataset.FlushCache()
    return dataset


def get_mask(nda: np.ndarray, min_mask_size = 0.02) -> np.ndarray:
    """
    Returns the mask as the smallest value of the dataset if the mask is bigger than the minimum size.

    :param nda: 2D array
    :param min_mask_size: minimum size of the mask in percentage of the total size (default 2%)
    :return: mask of the dataset if the mask is bigger than the minimum size (to avoid non-existing masks)
    """
    mask_value = np.min(nda)
    if np.count_nonzero(nda == mask_value) > int(nda.size * min_mask_size):
        return nda == mask_value
    return np.zeros_like(nda, dtype=bool)


def remove_mask_values(nda: np.ndarray, min_value_ratio=0.02) -> np.ndarray:
    """
    Removes the lowest value of the array and replaces it with NaN
    Used to display a dsm
    
    :param nda: array of the dataset
    :param min_value_ratio: minimum size of the mask in percentage of the total size (default 2%)
    :return: array of the dataset with the mask applied (mask values are set to NaN)
    """
    mask = get_mask(nda, min_value_ratio)
    nda[mask] = np.nan
    return nda


def to_mm(nda: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Converts from meters to millimeters
    ~ assuming the tallest building on earth is (830m) on top of mount everest (8848m)
    ~ 9,679,000mm -> fits in 32 signed bits (to keep negative values)
    
    :param nda: np.ndarray of shape (n, m).
    :param dtype: data type of the output array (default np.float32).
    :return: np.ndarray of shape (n, m) with values in millimeters.
    """
    return np.round(nda * 1000, 0).astype(dtype)


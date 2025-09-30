# copy paste augmentation

import numpy as np
from skimage.measure import label

from .nda import rotate as nda_rotate
from .nda import rescale_linear as nda_rescale_linear


def tight_crop(array) -> np.ndarray:
    """
    Crop around the non-zero values of the array.
    0 values are considered as background.
    
    :param array: Input array to be cropped.
    :return: Cropped array.
    """
    coords = np.argwhere(array != 0)
    minx = coords[:, 0].min()
    maxx = coords[:, 0].max()
    miny = coords[:, 1].min()
    maxy = coords[:, 1].max()
    return array[minx:maxx+1, miny:maxy+1]


def pad_to_tile(array: np.ndarray, *, angle=0.0, zoom=0.0, x=1000, y=1000, tile_size: int = 2000) -> np.ndarray:
    """
    Pad the input array to the specified tile size.  
    
    :param array: Input array to be padded.
    :param tile_size: Desired tile size (default is 2000).
    :param x: center x coordinate of the array in the tile
    :param y: center y coordinate of the array in the tile
    :return: Padded array.
    """
    array, _ = nda_rotate(array, angle)
    shape = (np.array(array.shape[:2]) * (1 + zoom)).astype(int)
    array = nda_rescale_linear(array, shape)

    y_size = array.shape[0]
    x_size = array.shape[1]

    w_dist = x - (x_size // 2)
    e_dist = tile_size - (x + (x_size - x_size // 2))
    n_dist = y - (y_size // 2)
    s_dist = tile_size - (y + (y_size - y_size // 2))

    if w_dist < 0:
        array = array[:, -w_dist:]
        w_dist = 0
        e_dist = tile_size - array.shape[1]

    if e_dist < 0:
        array = array[:, :e_dist]
        e_dist = 0
        w_dist = tile_size - array.shape[1]

    if n_dist < 0:
        array = array[-n_dist:, :]
        n_dist = 0
        s_dist = tile_size - array.shape[0]

    if s_dist < 0:
        array = array[:s_dist, :]
        s_dist = 0
        n_dist = tile_size - array.shape[0]

    padding = ((n_dist, s_dist), (w_dist, e_dist))
    if len(array.shape) == 3:
        padding = ((n_dist, s_dist), (w_dist, e_dist), (0, 0))
    
    array = np.pad(array, padding, mode='constant', constant_values=0)
    return array


def get_biggest_mask(masks: np.array) -> np.ndarray:
    """
    Gives the biggest mask that is not the background (0).  
    Returns empty mask if no masks are found.  

    :param masks: Input mask array with integer labels.
    :return: Boolean array where the biggest mask is True and others are False.
    """
    masks = label(masks)
    areas = [np.sum(masks == i) for i in range(1, masks.max() + 1)]
    if len(areas) == 0:
        return np.zeros_like(masks, dtype=bool)
    biggest_mask = np.argmax(areas) + 1
    return (masks == biggest_mask)


def get_copy_paste_mask(copy_array: np.ndarray, copy_ndsm: np.ndarray, paste_ndsm: np.ndarray) -> np.ndarray:
    """
    Get the mask of the visible part of the object in `copy_ndsm`  
    The resulting mask might be empty if the object is hidden under another  
    All 3 input arrays must have the same shape (except for depth of `copy_array`)

    :param copy_array: the image (orthophoto) of the object
    :param copy_ndsm: the masked ndsm of the object
    :param paste_ndsm: the ndsm of the tile where the object will be pasted (unmasked ndsm)
    :return: boolean mask of the visible part of the object in `copy_ndsm`
    """
    copy_mask = np.all(copy_array != 0, axis=-1)
    max_ndsm = np.maximum(copy_ndsm, paste_ndsm)
    mask = (max_ndsm == copy_ndsm) * copy_mask
    mask = get_biggest_mask(mask)
    return mask


def copy_paste(copy_array: np.ndarray, copy_ndsm: np.ndarray, paste_array: np.ndarray, paste_ndsm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Copy-paste augmentation of orthophoto and ndsm arrays.
    The `copy_array` and `copy_ndsm` are pasted onto the `paste_array` and `paste_ndsm`
    The pasted object will be visible only if it is above the `paste_ndsm` at that location.
    All 4 input arrays must have the same shape (except for depth of `copy_array` and `paste_array`)

    :param copy_array: the image (orthophoto) of the object to be copied
    :param copy_ndsm: the masked ndsm of the object to be copied
    :param paste_array: the image (orthophoto) of the tile where the object will be pasted
    :param paste_ndsm: the ndsm of the tile where the object will be pasted
    :return: tuple of (new orthophoto array, new ndsm array) after copy-paste
    """
    copy_mask = get_copy_paste_mask(copy_array, copy_ndsm, paste_ndsm)

    copy_array = (copy_array * np.dstack([copy_mask]*3))
    copy_ndsm = (copy_ndsm * copy_mask)
    
    paste_array = (paste_array * ~np.dstack([copy_mask]*3))
    paste_ndsm = (paste_ndsm * np.logical_not(copy_mask))

    return (copy_array + paste_array), (copy_ndsm + paste_ndsm)


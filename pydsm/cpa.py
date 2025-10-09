# copy paste augmentation

import numpy as np
from skimage.measure import label
from skimage.morphology import disk, binary_erosion
from scipy.ndimage import median_filter

from .nda import rotate as nda_rotate
from .nda import rescale_linear as nda_rescale_linear

from .tile import Tile


def is_mask_touching_border(mask: np.ndarray) -> bool:
    """
    Check if a binary mask is touching the border of the array.

    :param mask: 2D numpy array (binary mask)
    :return: True if the mask touches the border, False otherwise
    """
    if np.any(mask[0, :]) or np.any(mask[-1, :]):
        return True
    if np.any(mask[:, 0]) or np.any(mask[:, -1]):
        return True
    return False


def get_tight_crop_values(array: np.ndarray) -> tuple:
    """
    Gets the coordinates to tight crop around the mask in the array  

    :param array: np.ndarray, crop around any non-zero values
    :return: tuple, (min_x, max_x, min_y, max_y)
    """
    coords = np.argwhere(array != 0)
    minx = coords[:, 0].min()
    maxx = coords[:, 0].max()
    miny = coords[:, 1].min()
    maxy = coords[:, 1].max()
    return (minx,maxx+1,miny,maxy+1)


def extract_instances(tile: Tile, ignore_border=True) -> list[Tile]:
    """
    Extract the masked items from the copy_array and copy_ndsm based on the copy_labels.
    The background label (0) is ignored.

    :param tile_dict: dict [str, np.array] with keys `orthophoto`, `ndsm`, `instance_labels` and `semantic_labels`
    :param ignore_border: does not return the local tile of instances touching the border if True
    :return: a list of dict of [str, np.array] with keys `orthophoto`, `ndsm`, `instance_labels` and `semantic_labels`
    """
    local_tiles = []

    for v in np.unique(tile.instance_labels):
        if v == 0: continue  # Skip background

        mask = (tile.instance_labels == v)
        mask_rgb = np.dstack([mask] * 3)

        if ignore_border and is_mask_touching_border(mask):
            continue

        min_x, max_x, min_y, max_y = get_tight_crop_values(mask)

        ortho = (tile.orthophoto * mask_rgb)[min_x:max_x, min_y:max_y]
        ndsm = (tile.ndsm * mask)[min_x:max_x, min_y:max_y]
        instances = (tile.instance_labels * mask)[min_x:max_x, min_y:max_y]
        semantics = (tile.semantic_labels * mask)[min_x:max_x, min_y:max_y]

        local_tiles.append(Tile(ortho, ndsm, instances, semantics))
    
    return local_tiles


def pad_to_tile(array: np.ndarray, *, angle=0.0, zoom=0.0, x=1000, y=1000, tile_size: int = 2000) -> np.ndarray:
    """
    Pad the input array to the specified tile size.  
    
    :param array: Input array to be padded.
    :param tile_size: Desired tile size (default is 2000).
    :param x: center x coordinate of the array in the tile
    :param y: center y coordinate of the array in the tile
    :return: Padded array.
    """
    array = nda_rotate(array, angle)
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
    mask = median_filter(mask.astype(np.uint8), size=3).astype(bool)
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
    ndsm_mask = get_copy_paste_mask(copy_array, copy_ndsm, paste_ndsm)
    array_mask = binary_erosion(ndsm_mask, disk(2))

    copy_array = (copy_array * np.dstack([array_mask]*3))
    copy_ndsm = (copy_ndsm * ndsm_mask)
    
    paste_array = (paste_array * ~np.dstack([array_mask]*3))
    paste_ndsm = (paste_ndsm * np.logical_not(ndsm_mask))

    return (copy_array + paste_array), (copy_ndsm + paste_ndsm)


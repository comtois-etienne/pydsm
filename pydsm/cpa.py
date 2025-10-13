# copy paste augmentation

import numpy as np
from skimage.morphology import disk, binary_erosion, binary_dilation
from scipy.ndimage import median_filter

from .nda import rotate as nda_rotate
from .nda import rescale_linear as nda_rescale_linear
from .nda import rescale_nearest_neighbour as nda_rescale_nearest_neighbour
from .nda import is_mask_touching_border as nda_is_mask_touching_border
from .nda import get_biggest_mask as nda_get_biggest_mask
from .nda import remove_holes as nda_remove_holes
from .nda import is_mask_inside as nda_is_mask_inside

from .tile import Tile
from .tile import remove_small_masks
from .tile import flip_tile


def get_tight_crop_values(array: np.ndarray) -> tuple:
    """
    Gets the coordinates to tight crop around the mask in the array  
    Crop with `array[min_x:max_x, min_y:max_y]`  

    :param array: np.ndarray, crop around any non-zero values
    :return: tuple, ( min_x, max_x, min_y, max_y )
    """
    coords = np.argwhere(array != 0)
    min_x = coords[:, 0].min()
    max_x = coords[:, 0].max()
    min_y = coords[:, 1].min()
    max_y = coords[:, 1].max()
    return ( min_x, max_x+1, min_y, max_y+1 )


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

        if ignore_border and nda_is_mask_touching_border(mask):
            continue

        min_x, max_x, min_y, max_y = get_tight_crop_values(mask)

        ortho = (tile.orthophoto * mask_rgb)[min_x:max_x, min_y:max_y]
        ndsm = (tile.ndsm * mask)[min_x:max_x, min_y:max_y]
        instances = (tile.instance_labels * mask)[min_x:max_x, min_y:max_y]
        semantics = (tile.semantic_labels * mask)[min_x:max_x, min_y:max_y]

        local_tiles.append(Tile(ortho, ndsm, instances, semantics))
    
    return local_tiles


def rotate_local_tile(tile: Tile, angle: float) -> Tile:
    """
    Rotate a tile with only one instance.  
    Resulting tile is bigger or equal in size as the original due to the non-cropping rotation.  

    :param tile: Tile, with one instance
    :param angle: float, rotation in degree
    :return: Tile, with clockwise rotation of `angle` degrees. 
    """
    ortho = nda_rotate(tile.orthophoto, angle)
    ndsm = nda_rotate(tile.ndsm, angle)
    instance_labels = nda_rotate(tile.instance_labels, angle).astype(bool).astype(int)
    semantic_labels = instance_labels * np.max(tile.semantic_labels)

    return Tile(ortho, ndsm, instance_labels, semantic_labels)


def zoom_local_tile(tile: Tile, zoom=1.0):
    """
    Scale up or down a local tile (tile with only one instance)

    :param tile: Tile, tile with only one instance to change size
    :param zoom: float, `1.0` for no change max `1.9`, min `0.1`
    """
    zoom = max(min(1.9, zoom), 0.1)
    shape = (np.array(tile.ndsm.shape[:2]) * zoom).astype(int)

    ortho = nda_rescale_linear(tile.orthophoto, shape)
    ndsm = nda_rescale_linear(tile.ndsm, shape)
    instances = nda_rescale_nearest_neighbour(tile.instance_labels, shape).astype(bool).astype(int)
    semantics = instances * np.max(tile.semantic_labels)

    return Tile(ortho, ndsm, instances, semantics)


def pad_local_tile(tile: Tile, x=0, y=0, tile_size=2000):
    """
    Add zero padding to the local tile to allow it to be placed on the full tile

    :param tile: Tile, local tile with one instance
    :param x: int, position to place the `tile` into the full tile
    :param y: int, position to place the `tile` into the full tile
    :param tile_size: int, full tile size
    """
    ortho = tile.orthophoto
    ndsm = tile.ndsm
    instances = tile.instance_labels
    semantics = tile.semantic_labels

    y_size = tile.instance_labels.shape[0]
    x_size = tile.instance_labels.shape[1]

    w_dist = x - (x_size // 2)
    e_dist = tile_size - (x + (x_size - x_size // 2))
    n_dist = y - (y_size // 2)
    s_dist = tile_size - (y + (y_size - y_size // 2))

    if w_dist < 0:
        ortho = ortho[:, -w_dist:]
        ndsm = ndsm[:, -w_dist:]
        instances = instances[:, -w_dist:]
        semantics = semantics[:, -w_dist:]

        w_dist = 0
        e_dist = tile_size - instances.shape[1]

    if e_dist < 0:
        ortho = ortho[:, :e_dist]
        ndsm = ndsm[:, :e_dist]
        instances = instances[:, :e_dist]
        semantics = semantics[:, :e_dist]

        e_dist = 0
        w_dist = tile_size - instances.shape[1]

    if n_dist < 0:
        ortho = ortho[-n_dist:, :]
        ndsm = ndsm[-n_dist:, :]
        instances = instances[-n_dist:, :]
        semantics = semantics[-n_dist:, :]

        n_dist = 0
        s_dist = tile_size - instances.shape[0]

    if s_dist < 0:
        ortho = ortho[:s_dist, :]
        ndsm = ndsm[:s_dist, :]
        instances = instances[:s_dist, :]
        semantics = semantics[:s_dist, :]

        s_dist = 0
        n_dist = tile_size - instances.shape[0]

    padding = ((n_dist, s_dist), (w_dist, e_dist))
    padding_ortho = ((n_dist, s_dist), (w_dist, e_dist), (0, 0))

    ortho = np.pad(ortho, padding_ortho, mode='constant', constant_values=0)
    ndsm = np.pad(ndsm, padding, mode='constant', constant_values=0)
    instances = np.pad(instances, padding, mode='constant', constant_values=0)
    semantics = np.pad(semantics, padding, mode='constant', constant_values=0)

    return Tile(ortho, ndsm, instances, semantics)


def get_copy_paste_tile(copy_tile: Tile, paste_tile: Tile, remove_cracks=5) -> Tile:
    """
    Get the mask of the visible part of the ndsm in copy_tile  
    The resulting mask might be empty if the object is hidden under another  
    The 2 tiles must have the same shape

    :param copy_tile: Tile, a tile with only one instance. same shape as paste_tile  
    :param paste_ndsm: Tile, a tile with or without multiple instances.  
    :param remove_cracks: int, the crack size in the masks to remove (caused by the intersection of the ndsm)  
    :return: Tile, containing the visible part of copy_tile when intersected with paste_tile
    """
    max_ndsm = np.maximum(copy_tile.ndsm, paste_tile.ndsm)
    mask = (max_ndsm == copy_tile.ndsm) * copy_tile.instance_labels
    if remove_cracks:
        mask = binary_dilation(mask, disk(remove_cracks)) * copy_tile.instance_labels
    mask = nda_get_biggest_mask(mask)
    mask = median_filter(mask.astype(np.uint8), size=3).astype(bool)
    mask = nda_remove_holes(mask)

    ortho = copy_tile.orthophoto * np.dstack([mask] * 3)
    ndsm = copy_tile.ndsm * mask
    semantics = mask * np.max(copy_tile.semantic_labels)

    return Tile(ortho, ndsm, mask, semantics)


def copy_paste(copy_tile: Tile, paste_tile: Tile, remove_cracks=5, remove_masks=400) -> Tile:
    """
    Copy-paste augmentation of RGBD tiles with instance and semantic segmentation  
    Both tiles have to be the same shape. 

    :param copy_tile: Tile, containing one instance with one semantic label
    :param paste_tile: Tile, can contain many instances with multiple semantic labels
    :param remove_cracks: int, the crack size in the masks to remove
    :param remove_masks: int, the size of the masks to be removed (smaller or equal to)
    :return: Tile, copy-pasted tile 
    """
    copy_tile = get_copy_paste_tile(copy_tile, paste_tile, remove_cracks)
    ndsm_mask = copy_tile.instance_labels.astype(bool)
    array_mask = binary_erosion(ndsm_mask, disk(2))
    
    # orthophoto
    copy_ortho = (copy_tile.orthophoto * np.dstack([array_mask]*3))
    paste_ortho = (paste_tile.orthophoto * ~np.dstack([array_mask]*3))
    ortho = (copy_ortho + paste_ortho)
    
    # ndsm
    copy_ndsm = (copy_tile.ndsm * ndsm_mask)
    paste_ndsm = (paste_tile.ndsm * np.logical_not(ndsm_mask))
    ndsm = (copy_ndsm + paste_ndsm)

    # instances
    instance_val = np.max(paste_tile.instance_labels) + 1
    instances = paste_tile.instance_labels * ~ndsm_mask
    instances += (ndsm_mask * instance_val)

    # semantics
    semantics = paste_tile.semantic_labels * ~ndsm_mask
    semantics += copy_tile.semantic_labels

    return remove_small_masks(Tile(ortho, ndsm, instances, semantics), remove_masks)


def random_copy_paste(copy_local_tile: Tile, paste_tile: Tile, dim_change=0.1, overlap_ratio=0.3, remove_cracks=5, remove_masks=400) -> Tile:
    """
    Copy-paste a single instance to a Tile 

    :param copy_local_tile: Tile, containing a single instance. Size must be smaller than `paste_tile`
    :param paste_tile: Tile, to paste the `copy_local_tile` onto
    :param dim_change: float, plus-minus scale value of the pasted instance
    :param overlap_ratio: float, the maximum overlap between pasted instances
    :param remove_cracks: int, the size of the cracks to be removed
    :param remove_masks: int, the size of the masks to be removed (smaller or equal to)
    :return: Tile, with pasted instance
    """
    tile_size = paste_tile.ndsm.shape[0]
    angle = np.random.uniform(0, 360)
    x = np.random.randint(0, tile_size)
    y = np.random.randint(0, tile_size)
    zoom = 1 + np.random.uniform(-dim_change, dim_change)

    copy_local_tile = flip_tile(copy_local_tile, axis=0) if np.random.rand() > 0.5 else copy_local_tile
    copy_local_tile = flip_tile(copy_local_tile, axis=1) if np.random.rand() > 0.5 else copy_local_tile
    copy_local_tile = zoom_local_tile(copy_local_tile, zoom)
    copy_local_tile = rotate_local_tile(copy_local_tile, angle)
    copy_tile = pad_local_tile(copy_local_tile, x, y, tile_size)

    is_inside = nda_is_mask_inside(
        copy_tile.instance_labels, 
        paste_tile.instance_labels, 
        inside_ratio=overlap_ratio
    )

    if is_inside: return paste_tile
    return copy_paste(copy_tile, paste_tile, remove_cracks, remove_masks)


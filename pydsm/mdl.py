import os
import osgeo
import numpy as np
import pandas as pd
import imageio.v3 as iio
from skimage.measure import label
from skimage.morphology import disk
import tensorflow as tf
from typing import Callable
from scipy.ndimage import median_filter

import pydsm.nda as nda
import pydsm.geo as geo
import pydsm.utils as utils


def training_split(ortho_path, ndms_path, mask_path, output_dir, scale=0.02, overlap=0.5, max_empty=0.9, size=1024, save_png=True):
    overlap = min(0.99, overlap)
    max_empty = min(0.99, max_empty)

    uuid = utils.remove_extension(utils.get_filename(ortho_path))
    uuid = uuid.split('_')[0]

    # open orthophoto
    ortho = geo.open_geotiff(ortho_path)
    ortho_scale = geo.geo_get_scales(ortho)[0]
    ortho_diff = abs(1 - ortho_scale / scale)

    # open ndsm
    ndsm = geo.open_geotiff(ndms_path)
    ndsm_scale = geo.get_scales(ndsm)[0]
    ndsm_diff = abs(1 - ndsm_scale / scale)

    # open mask
    mask = nda.read_numpy(mask_path, npz_format='napari')

    if ortho_diff > 0.001: ortho = geo.rescale(ortho, scale)
    if ndsm_diff > 0.001: ndsm = geo.rescale(ndsm, scale)

    ortho = geo.to_ndarray(ortho)
    ndsm = geo.to_ndarray(ndsm)
    mask = nda.rescale_nearest_neighbour(mask, ortho.shape[:2])

    ortho_iterator = nda.SubimageGenerator(ortho, size=size, min_overlap=overlap)
    ndsm_iterator = nda.SubimageGenerator(ndsm, size=size, min_overlap=overlap)
    mask_iterator = nda.SubimageGenerator(mask, size=size, min_overlap=overlap)

    while True:
        try:
            coord, subimage = next(ortho_iterator)
            _, subndsm = next(ndsm_iterator)
            _, submask = next(mask_iterator)
        except StopIteration:
            break

        # Dont save the subimage if the empty region is larger than the overlap
        perc_zero = 1 - np.count_nonzero(subimage[:, :, 3]) / (subimage.size // 4)
        if perc_zero > max_empty: continue

        center = np.array(submask.shape[:2]) // 2
        radius = int((size / 1.414) / 1.414) // 2 - 100
        center_sub = submask[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius]

        if np.all(center_sub == 0):
            print(f'Skipping {uuid}_{coord[0]}_{coord[1]}: empty submask')
            continue
        
        subimage = subimage[:, :, :3]
        subimage = nda.normalize(subimage)
        submask = label(submask) # todo can cause problems during training, should be fixed

        name = f'{uuid}_{coord[0]}_{coord[1]}.npz'
        np.savez_compressed(
            os.path.join(output_dir, name),
            orthophoto=subimage,
            ndsm=subndsm,
            mask=submask,
            y=coord[0],
            x=coord[1],
            uuid=uuid,
            scale=scale,
            overlap=overlap,
        )

        if save_png:
            img = nda.to_uint8(subimage)
            img = img.astype(np.uint8)
            iio.imwrite(
                os.path.join(output_dir, f'{uuid}_{coord[0]}_{coord[1]}.png'),
                img,
            )


def get_tree_species_dict() -> dict:
    """
    arrière-plan + inconnu + 5 especes + 13 genres + 1 famille
    21 classes total
    """
    UNKNOWN_TREE = 1
    BETULACEAE = 17
    return {
        'BACKGROUND': 0,
        'UNKNOWN_TREE': UNKNOWN_TREE,
        'ACSA': 2,
        'ACPL': 3,
        'ACSC': 4,
        'ULWI': 5,
        'ULXX': 6,
        'UL': 7,
        'PO': 8,
        'FR': 9,
        'TI': 10,
        'QU': 11,
        'AM': 12,
        'PI': 13,
        'JU': 14,
        'GL': 15,
        'PN': 16,
        'BE': BETULACEAE,
        'OS': BETULACEAE,
        'AL': BETULACEAE,
        'CA': BETULACEAE,
    }


def get_tree_species_code(species: str) -> int:
    tree_species = get_tree_species_dict()
    species_long = species.upper()[:4]
    species_short = species_long[:2]

    if species_long in tree_species:
        return tree_species[species_long]
    
    if species_short in tree_species:
        return tree_species[species_short]

    return tree_species['UNKNOWN_TREE']


def apply_tree_species_codes(labels: np.ndarray, trees_df: pd.DataFrame, code_replacement_function: Callable = get_tree_species_code) -> np.ndarray:
    """
    :param array: numpy array, dtype must be integer. contains the mask instances of the trees
    :param trees_df: pandas DataFrame containing the tree position with their species (must contain ['axis-0', 'axis-1', 'code'])
    :param code_replacement_function: function that takes a species string and returns an integer species code
    :return: np.ndarray with the species codes applied to the array
    """
    unique_values = np.unique(labels)
    unique_values = unique_values[1:] if unique_values[0] == 0 else unique_values

    trees = trees_df[['axis-0', 'axis-1', 'code']].copy()
    trees.rename(columns={'axis-0': 'y', 'axis-1': 'x', 'code': 'species'}, inplace=True)

    trees['v'] = 0
    for i, (y, x) in enumerate(zip(trees['y'], trees['x'])):
        row_indexer = trees.index == i
        trees.loc[row_indexer, 'v'] = labels[y, x]
    trees = trees[trees['v'] != 0]
    trees = trees.drop(columns=['x', 'y'])

    missing_species = []
    for v in unique_values:
        if v not in trees['v'].values:
            missing_species.append(v)

    for v in missing_species:
        df = pd.DataFrame({'species': '', 'v': v}, index=[0])
        trees = pd.concat([trees, df], ignore_index=True)

    trees['species_code'] = 0
    for species in trees['species']:
        code = code_replacement_function(species)
        row_indexer = trees['species'] == species
        trees.loc[row_indexer, 'species_code'] = code

    v_list = trees['v'].tolist()
    s_list = trees['species_code'].tolist()

    return nda.replace_value_inplace(labels, v_list, s_list)


def downsample_species_array(array_species: np.ndarray, size=100) -> np.ndarray:
    from scipy.ndimage import median_filter
    array_species_downsampled = nda.rescale_nearest_neighbour(array_species, (size, size))
    array_species_downsampled = median_filter(array_species_downsampled, size=3)
    return array_species_downsampled


def preprocess_tile(orthophoto: np.ndarray, dsm: np.ndarray, labels: np.ndarray, points: pd.DataFrame, downsampling_size=100, dsm_clip_height=30.0) -> np.ndarray:
    """
    :param orthophoto: np.ndarray with shape (H, W, 4) or (H, W, 3) representing the orthophoto image.
    :param dsm: np.ndarray with shape (H, W) representing the digital surface model (DSM or NDSM).
    :param labels: np.ndarray with shape (H, W) representing the instance segmentation labels.
    :param points: pd.DataFrame with columns `['axis-0', 'axis-1', 'code']` representing the tree species points.
    :param downsampling_size: int, size to which the labels will be downsampled.
    :param dsm_clip_height: float, height to which the DSM will be clipped.
    :return: dict containing the processed `{orthophoto, dsm, labels, labels_downsampled, species, centers, centers_downsampled}`.  
        - `orthophoto` is normalized to `[0.0, 1.0]` range  
        - `dsm` is clipped and rescaled to `[0.0, 1.0]` range where `1.0` equals `dsm_clip_height`
        - `labels` are relabeled to have unique values starting from `0` (does not use skimage's label function)  
        - `species` is a numpy array with the tree species codes `(int)` applied to the labels  
        - `centers` is a binary numpy array with the centers of the instances set to `1.0`
        - `centers_downsampled` is a numpy array with the centers of the downsampled instances
    """
    orthophoto = nda.normalize(orthophoto[..., :3])
    dsm = dsm[..., :1] if dsm.ndim > 2 else dsm
    dsm = nda.clip_rescale(dsm, dsm_clip_height)
    labels = nda.relabel(labels)
    species = apply_tree_species_codes(labels, points, get_tree_species_code)
    
    labels_downsampled = nda.rescale_nearest_neighbour(labels, (downsampling_size, downsampling_size))
    centers = np.array(nda.get_labels_centers(labels_downsampled))

    species_downsampled = nda.rescale_nearest_neighbour(species, (downsampling_size, downsampling_size))
    species_downsampled = median_filter(species_downsampled, size=3)

    array_centers_downsampled = np.zeros_like(labels_downsampled)
    array_centers_downsampled[centers[:, 0], centers[:, 1]] = 1

    centers = centers * (orthophoto.shape[0] // downsampling_size)
    array_centers = np.zeros_like(labels)
    array_centers[centers[:, 0], centers[:, 1]] = 1

    labels_downsampled = median_filter(labels_downsampled, size=3)

    return {
        'orthophoto': orthophoto,
        'dsm': dsm,
        'labels': labels,
        'labels_downsampled': labels_downsampled,
        'species': species,
        'species_downsampled': species_downsampled,
        'centers': array_centers,
        'centers_downsampled': array_centers_downsampled
    }



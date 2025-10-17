from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.ndimage import median_filter
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from cv2 import resize as cv2_resize
from cv2 import INTER_CUBIC

import pydsm.nda as nda
import pydsm.geo as geo
import pydsm.utils as utils


@dataclass
class Tile:
    orthophoto: np.ndarray
    ndsm: np.ndarray
    instance_labels: np.ndarray
    semantic_labels: np.ndarray

    def copy(self) -> 'Tile':
        return Tile(
            self.orthophoto.copy(), 
            self.ndsm.copy(), 
            self.instance_labels.copy(), 
            self.semantic_labels.copy()
        )
    
    def rgbd(self, norm = False) -> np.ndarray:
        rgb = nda.to_uint8(self.orthophoto, norm=norm)
        d = nda.to_uint8(self.ndsm, norm=norm)
        return np.dstack((rgb, d))


def default_semantic_dict() -> dict:
    return {
        'BACKGROUND': 0,
        'UNKNOWN': 1,
    }


def tree_species_dict_v1() -> dict:
    """
    arrière-plan + inconnu + 5 especes + 10 genres + 1 famille
    18 classes total
    """
    return {
        'BACKGROUND': 0,
        'UNKNOWN_TREE': 1,
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
        'BE': 17,
        'OS': 17,
        'AL': 17,
        'CA': 17,
        # 'SY': 18,
    }


def tree_species_dict_v2() -> dict:
    """
    20 classes total : 18 espèces + 1 arrière-plan + 1 inconnu
    """
    return {
        'BACKGROUND': 0,
        'UNKNOWN': 1,
        'ACSA': 2,
        'ACPL': 3,
        'ACSC': 4,
        'UL' : 5, # split? ULWI, ULXX, UL
        'FR': 6, # split?
        'QU': 7,
        'PO': 8,
        'GL': 9,
        'SY': 10,
        'PI': 11,
        'CE': 12,
        'TI': 13,
        'AM': 14,
        'GY': 15,
        'GI': 16,
        'MA': 17,
        'PN': 18,
        'JU': 19,
    }


def get_semantic_code(semantics_dict: dict, code: int) -> str:
    """
    `Warning` does not check for out of bound errors

    :param semantics_dict: dict[str, int] that maps semantic str to semantic int
    :param code: int, int value of the semantic str
    :return: str, semantic str
    """
    return list(semantics_dict)[code]


def get_semantic_intcode(semantics_dict: dict, code: str) -> int:
    """
    Get the integer value of the given code  
    4 first char are used. If no corresponding value was found, the 2 first char are used  
    Returns semantics_dict['UNKNOWN'] if nothing is found  

    :param semantics_dict: dict[str, int], semantic string mapped to an integer value
    :param code: str, key value with corresponding int in the `semantics_dict`
    :return: int value corresponding to the code string
    """
    long = code.upper()[:4]
    short = long[:2]

    if long in semantics_dict:
        return semantics_dict[long]
    elif short in semantics_dict:
        return semantics_dict[short]
    else:
        return semantics_dict['UNKNOWN']


def get_all_points(tiles_dir: str):
    """
    Loads all annotated points from CSV files in the 'points' subdirectory of tiles_dir.  
    Concatenates them into a single DataFrame.

    :param tiles_dir: Directory containing a 'points' subdirectory with CSV files.
    :return: DataFrame containing all points from the CSV files.
    """
    points_dir = os.path.join(tiles_dir, 'points')
    
    all_points = None
    for name in os.listdir(points_dir):
        if not name.endswith('.csv'): continue
        csv = pd.read_csv(os.path.join(points_dir, name))
        if len(csv) == 0: continue

        if all_points is None:
            all_points = csv
        else:
            all_points = pd.concat([all_points, csv], ignore_index=True)

    return all_points


def get_semantic_count(all_points: pd.DataFrame, semantic_dict: dict, semantic_col: str = 'code'):
    """
    Return the count of each semantic class in the points DataFrame.  
    The semantic classes are defined in the semantic_dict.  
    Class Background (int_code 0) and Unknow (int_code 1) are always added with a count of 0.  

    :param all_points: DataFrame containing the points with a semantic column.
    :param semantic_dict: Dictionary mapping semantic codes to integer codes.
    :param semantic_col: Name of the column in all_points that contains the semantic codes.
    :return: List of counts for each semantic class, ordered by the integer codes.
    """
    all_points = all_points[[semantic_col]].copy()
    all_points['int_code'] = all_points[semantic_col].apply(
        lambda code: get_semantic_intcode(semantic_dict, code)
    )
    occurrences = all_points['int_code'].value_counts().sort_index()
    # occurrences.index = occurrences.index.map(lambda int_code: tile.get_semantic_code(semantic_dict, int_code))
    # print(occurrences)
    return [0, 0] + occurrences.to_list()[1:]


def get_semantic_distribution(tiles_dir: str, semantic_dict: dict) -> list:
    """
    Return the distribution of semantic classes in the points DataFrame.  
    The semantic classes are defined in the semantic_dict.  

    :param tiles_dir: str, directory containing a 'points' subdirectory with CSV files.
    :param semantic_dict: dict, mapping semantic codes to integer codes.
    :return: list, of the distribution of each semantic class, ordered by the integer codes as indexes.
    """
    all_points = get_all_points(tiles_dir)
    semantic_count = get_semantic_count(all_points, semantic_dict, 'code')
    semantic_count = np.array(semantic_count) / np.sum(semantic_count)
    return semantic_count.tolist()


def get_augmentation_distribution(distribution: list) -> list:
    """
    Gives the inverse of the distribution, normalized to sum to 1.  
    This can be used to select instances to augment in order to rebalance the dataset.
    Zero values are kept to zero.

    :param distribution: list, of the distribution of each semantic class, ordered by the integer codes as indexes.
    :return: list, of the inverse distribution, normalized to sum to 1.
    """
    distribution = np.array(distribution)
    max_val = np.max(distribution) * 2
    inverse = np.full_like(distribution, max_val)
    inverse[distribution == 0.0] = 0.0
    inverse -= distribution
    return inverse / np.sum(inverse)


def display_tile(tile: Tile, colorbar=False, semantic_dict=tree_species_dict_v2(), instance_cmap='tab20b', semantic_cmap='tab20'):
    plt.subplots(1, 4, figsize=(20, 10))

    plt.subplot(1, 4, 1)
    plt.title('orthophoto')
    plt.imshow(tile.orthophoto)

    plt.subplot(1, 4, 2)
    max_v = np.max(tile.ndsm)
    plt.title(f'ndsm={max_v:.1f}')
    plt.imshow(tile.ndsm, cmap='terrain')

    plt.subplot(1, 4, 3)
    unique = len(np.unique(tile.instance_labels))
    plt.title(f'instance_labels={unique}')
    plt.imshow(tile.instance_labels, cmap=instance_cmap, interpolation='nearest')

    plt.subplot(1, 4, 4)
    unique = len(np.unique(tile.semantic_labels))
    if unique == 2:
        unique = np.max(tile.semantic_labels)
        unique = get_semantic_code(semantic_dict, unique)

    plt.title(f'semantic_labels={unique}')
    plt.imshow(tile.semantic_labels, vmin=0, vmax=20, cmap=semantic_cmap, interpolation='nearest')
    plt.colorbar() if colorbar else None

    plt.show()


def apply_semantic_codes(instance_labels: np.ndarray, semantics_df: pd.DataFrame, semantics_dict: dict) -> np.ndarray:
    """
    Creates the semantic labels from the instance labels 

    :param instance_labels: numpy array, dtype must be integer. contains the mask instances
    :param semantics_df: pd.DataFrame with columns ['axis-0', 'axis-1', 'code'] representing the semantic labels.
    :param semantics_dict: dictionary mapping the semantic str (code) to integer values
    :return: np.ndarray with the semantic int values applied to the instance masks
    """
    unique_values = np.unique(instance_labels)
    unique_values = unique_values[1:] if unique_values[0] == 0 else unique_values

    df = semantics_df[['axis-0', 'axis-1', 'code']].copy()
    # df.rename(columns={'axis-0': 'y', 'axis-1': 'x', 'code': 'species'}, inplace=True)

    df['v'] = 0
    for i, (y, x) in enumerate(zip(df['axis-0'], df['axis-1'])):
        row_indexer = df.index == i
        df.loc[row_indexer, 'v'] = instance_labels[int(y), int(x)]
    df = df[df['v'] != 0]
    df = df.drop(columns=['axis-0', 'axis-1'])

    missing_codes = []
    for v in unique_values:
        if v not in df['v'].values:
            missing_codes.append(v)

    for v in missing_codes:
        missing_df = pd.DataFrame({'code': '', 'v': v}, index=[0])
        df = pd.concat([df, missing_df], ignore_index=True)

    df['int_code'] = 0
    for code in df['code']:
        int_code = get_semantic_intcode(semantics_dict, code)
        row_indexer = df['code'] == code
        df.loc[row_indexer, 'int_code'] = int_code

    v_list = df['v'].tolist()
    s_list = df['int_code'].tolist()

    return nda.replace_value_inplace(instance_labels, v_list, s_list)


def remove_holes(instance_labels: np.ndarray) -> np.ndarray:
    """
    Remove holes of any size from all instances  

    :param instance_labels: np.ndarray containing all instances
    :return: the instance_labels with per instance holes removal
    """
    new_instance_labels = np.zeros_like(instance_labels)
    for v in np.unique(instance_labels):
        mask = (instance_labels == v)
        mask = nda.remove_holes(mask)
        new_instance_labels += (mask * v)
    return new_instance_labels


def remove_small_masks(tile: Tile, min_area=400):
    """
    Removes all instances with an area lower or equal to `min_area`  
    The instances are relabeled to find unconnected parts of instances  

    :param tile: Tile, a tile with potentially small or unconnected parts of instances  
    :param min_area: int, rejects masks with lower or equal area (default=20*20=400)
    :return: Tile, a tile with modified (or not) instance_labels and semantic_labels
    """
    labeled = label(tile.instance_labels)
    for v in np.unique(labeled):
        mask = (labeled == v)
        if np.sum(mask) > min_area:
            continue
        tile = Tile(
            tile.orthophoto,
            tile.ndsm,
            tile.instance_labels * ~mask,
            tile.semantic_labels * ~mask,
        )
    return tile


def flip_tile(tile: Tile, axis=0):
    """
    Flip on axis 1 or 2

    :param tile: Tile, tile to be flipped
    :param axis: int, flip axis
    """
    return Tile(
        np.flip(tile.orthophoto, axis=axis),
        np.flip(tile.ndsm, axis=axis),
        np.flip(tile.instance_labels, axis=axis),
        np.flip(tile.semantic_labels, axis=axis),
    )


def open_tile(tiles_dir: str, tile_name: str, semantic_dict=default_semantic_dict(), as_array=True) -> Tile:
    """
    Opens all data from the tile (orthophoto, ndsm, instance_labels, semantic_labels)

    :param tiles_dir: str, directory containing the sub-directories `orthophoto`, `ndsm`, `labels` and `points`
    :param tile_name: str, tile name in the sub-directories
    :param as_array: bool, returns the orthophoto and the ndsm as numpy arrays if True, as gdal.Dataset if False
    :return: Tile containing `orthophoto`, `ndsm`, `instance_labels` and `semantic_labels`
    """
    tile_name = utils.remove_extension(tile_name)

    ortho = geo.open_geotiff(os.path.join(tiles_dir, 'orthophoto', f'{tile_name}.tif'))
    ndsm = geo.open_geotiff(os.path.join(tiles_dir, 'ndsm', f'{tile_name}.tif'))
    instances = nda.read_numpy(os.path.join(tiles_dir, 'labels', f'{tile_name}.npz'), npz_format='napari')
    instances = nda.relabel(instances)
    instances = remove_holes(instances)

    points_df = pd.read_csv(os.path.join(tiles_dir, 'points', f'{tile_name}.csv'))
    semantics = apply_semantic_codes(instances, points_df, semantic_dict)

    if as_array:
        ortho = geo.to_ndarray(ortho)[..., :3]
        ndsm = geo.to_ndarray(ndsm)

    return Tile(ortho, ndsm, instances, semantics)


def split_tile(tile: Tile) -> list[Tile]:
    """
    Splits a single tile into 4 identically sized tiles  

    :param tile: tile containing orthophoto, ndsm, instance labels, and semantic labels
    :returns: a list of 4 tiles (`top-left`, `top-right`, `bottom-left`, `bottom-right`)
    """
    h, w = tile.orthophoto.shape[:2]
    h, w = h // 2, w // 2

    sub_tiles = []
    for i in range(2):
        for j in range(2):
            ortho = tile.orthophoto[i * h:(i + 1) * h, j * w:(j + 1) * w]
            ndsm = tile.ndsm[i * h:(i + 1) * h, j * w:(j + 1) * w]
            instances = tile.instance_labels[i * h:(i + 1) * h, j * w:(j + 1) * w]
            semantics = tile.semantic_labels[i * h:(i + 1) * h, j * w:(j + 1) * w]
            sub_tiles.append(Tile(ortho, ndsm, instances, semantics))

    return sub_tiles


def normalize_ndsm(tile: Tile, clip_height=30.0) -> Tile:
    """
    Normalize the ndsm to [0.0, 1.0] range where 1.0 equals `clip_height`

    :param tile: Tile, with an ndsm to be normalized
    :param clip_height: float, height to which the DSM will be clipped.
    :return: Tile, tile with a normalized ndsm
    """
    ndsm = tile.ndsm[..., :1] if tile.ndsm.ndim > 2 else tile.ndsm
    ndsm = nda.clip_rescale(ndsm, clip_height)
    return Tile(tile.orthophoto, ndsm, tile.instance_labels, tile.semantic_labels)


def normalize_orthophoto(tile: Tile) -> Tile:
    """
    Normalize the orthophoto to [0.0, 1.0] range

    :param tile: Tile, with an orthophoto to be normalized
    :return: Tile, tile with a normalized orthophoto
    """
    orthophoto = nda.normalize(tile.orthophoto[..., :3])
    return Tile(orthophoto, tile.ndsm, tile.instance_labels, tile.semantic_labels)


def preprocess_tile(tile: Tile, ndsm_clip_height=30.0, min_mask_size=400) -> Tile:
    """
    Preprocess a tile so it can be used for training a model.  

    :param tile: tile containing orthophoto, ndsm, instance labels, and semantic labels
    :param dsm_clip_height: float, height to which the DSM will be clipped.
    :param min_mask_size: int, keep the masks of size greater than (default=400)
    :return: Tile containing the processed `{orthophoto, dsm, labels, labels_downsampled, species, centers, centers_downsampled}`.  
        - `orthophoto` is normalized to `[0.0, 1.0]` range  
        - `ndsm` is clipped and rescaled to `[0.0, 1.0]` range where `1.0` equals `dsm_clip_height`
    """
    tile = remove_small_masks(tile, min_mask_size)
    tile = normalize_orthophoto(tile)
    tile = normalize_ndsm(tile, ndsm_clip_height) if ndsm_clip_height > 0.0 else tile
    return tile


def correct_ndsm(tile: Tile, subsampling_size=8, median_kernel_size=3) -> Tile:
    """
    Correct the ndsm by replacing the aberant values

    :param tile: Tile, with an ndsm to be corrected
    :param subsampling_size: int, used to apply a median filter on a subsampled ndsm
    :param median_kernel_size: int, kernel size of the median filter applied before scaling
    :return: Tile, tile with a corrected (approximation) of the ndsm
    """
    instance_mask = tile.instance_labels.astype(bool)
    ndsm = tile.ndsm
    shape = ndsm.shape
    filled_dtm = ndsm.copy()

    mean = np.mean(ndsm[ndsm != 0])
    mask = (ndsm < (mean / 2))
    _, idx = distance_transform_edt(mask, return_indices=True)
    filled_dtm[mask] = ndsm[tuple(idx[:, mask])]

    dtm_simple = filled_dtm[::subsampling_size, ::subsampling_size]
    dtm_smoothed = median_filter(dtm_simple, size=median_kernel_size)

    dtm_upsampled = cv2_resize(dtm_smoothed, (shape[1], shape[0]), interpolation=INTER_CUBIC)
    dtm_upsampled = (dtm_upsampled + ndsm) / 2
    dtm_upsampled = np.maximum(ndsm, dtm_upsampled)
    dtm_upsampled = dtm_upsampled * instance_mask

    return Tile(tile.orthophoto, dtm_upsampled, tile.instance_labels, tile.semantic_labels)


def save_tile(npz_path: str, tile: Tile) -> None:
    """
    Saves a tile to disk

    :param npz_path: str, path to save the tile as a compressed numpy array
    :param tile: Tile, tile to save
    :return: None, save to disk
    """
    np.savez_compressed(
        npz_path,
        orthophoto=tile.orthophoto,
        ndsm=tile.ndsm,
        instance_labels=tile.instance_labels,
        semantic_labels=tile.semantic_labels,
    )


def open_tile_npz(npz_path: str) -> Tile:
    """
    Opens a tile that has been saved as a compressed numpy array

    :param npz_path: str, path to the tile
    :return: Tile from disk
    """
    npz = np.load(npz_path, allow_pickle=True)
    return Tile(npz['orthophoto'], npz['ndsm'], npz['instance_labels'], npz['semantic_labels'])


def get_instance(tiles_dir: str, semantic_code: str, percentile=0.0) -> Tile:
    """
    Get an instance of a given semantic class from the tiles directory.

    :param tiles_dir: str, Path to the tiles directory (should contain 'instances' subdirectory)
    :param semantic_code: str, Semantic code of the desired class (e.g., 'PI' for Pinus)
    :param percentile: float, Percentile to select the instance (0.0 for first, 0.5 for median, 1.0 for last)
    :return: Tile, at the specified percentile of the instances of the given semantic class.
    """
    percentile = max(0.0, min(0.999999, percentile))
    tile_files = sorted(os.listdir(os.path.join(tiles_dir, 'instances', semantic_code)))
    tile_files = [f for f in tile_files if f.endswith('.npz')]
    index = int(percentile * len(tile_files))
    tile_file = tile_files[index]
    tile_path = os.path.join(tiles_dir, 'instances', semantic_code, tile_file)
    return open_tile_npz(tile_path)


def get_random_instances(tiles_dir: str, semantic_dict: dict, distribution: list, size=1) -> list[Tile] | Tile:
    """
    Get a random instance from the tiles directory based on the given distribution.  
    The distribution dictactes the probability of selecting each semantic class.  

    :param tiles_dir: str, Path to the tiles directory (should contain 'instances' subdirectory)
    :param semantic_dict: dict, mapping semantic codes to integer codes.
    :param distribution: list, of the distribution of each semantic class, ordered by the integer
    :return: Tile | list[Tile], a random instance from the tiles directory based on the given distribution.
    """
    semantics = list(semantic_dict)
    codes = np.random.choice(semantics, size=size, p=distribution)
    instances = []
    for code in codes:
        percentile = np.random.uniform(0.0, 1.0)
        instance = get_instance(tiles_dir, code, percentile)
        instances.append(instance)
    return instances if size > 1 else instances[0]


def save_split_tiles(tiles_dir: str, tile_name: str, tiles: list[Tile]):
    """
    Saves the tiles into `tiles_dir` using tile_name with their orientation  
    Tiles in order '(nw)', '(ne)', '(sw)', '(se)'  

    :param tiles_dir: str, directory to save the tiles in
    :param tile_name: str, the name of the tile (extension will be replaced) 
    :param tiles: list[Tile], 4 tiles with orientation nw, ne, sw, and se
    :return: None, saves the tiles to disk
    """
    names = ['nw', 'ne', 'sw', 'se']
    tile_name = utils.remove_extension(tile_name)

    for i, tile in enumerate(tiles):
        save_path = f'{tile_name} ({names[i]}).npz'
        save_path = utils.append_file_to_path(tiles_dir, save_path)
        save_tile(save_path, tile)


def create_tile_dataset(tiles_dir: str, save_sub_dir: str = 'dataset', semantic_dict=default_semantic_dict(), split=True, normalize=True, min_mask_size=400) -> None:
    """
    Saves the annotated tiles to npz to be used for training  
    Tiles are normalized  
    Each tile is split into 4 'nw', 'ne', 'sw', 'se' sub-tiles  
    
    :param tiles_dir: str, directory containing the sub-directories `orthophoto`, `ndsm`, `labels`, and `points`
    :param save_sub_dir: sub dir to save the tiles in
    :return: None, saves the tiles into the sub-dir of tiles_dir
    """
    names = os.listdir(os.path.join(tiles_dir, 'orthophoto'))
    save_dir = os.path.join(tiles_dir, save_sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    for name in names:
        if not name.endswith('.tif'): continue
        tile = open_tile(tiles_dir, name, semantic_dict)
        if split:
            tiles = split_tile(tile)
            tiles = [remove_small_masks(tile, min_mask_size) for tile in tiles]
            tiles = [normalize_ndsm(tile) if normalize else tile for tile in tiles]
            tiles = [normalize_orthophoto(tile) if normalize else tile for tile in tiles]
            save_split_tiles(save_dir, name, tiles)
        else:
            file_name = f'{utils.remove_extension(name)}.npz'
            npz_path = utils.append_file_to_path(save_dir, file_name)
            tile = remove_small_masks(tile, min_mask_size)
            tile = normalize_ndsm(tile) if normalize else tile
            tile = normalize_orthophoto(tile) if normalize else tile
            save_tile(npz_path, tile)


def create_split_tile_dataset(tiles_dir: str, sub_dir: str, save_dir: str = 'dataset', normalize=True, min_mask_size=400):
    """
    Split all npz tiles from `tiles_dir/sub_dir` and saves them into `tiles_dir/save_dir`  

    :param tiles_dir: str, directory where the tiles are saved
    :param sub_dir: str, sub-directory where the tiles are saved
    :param save_dir: str, sub-directory where the split tiles will be saved
    :param normalize: bool, whether to normalize the nDSM and orthophoto
    :param min_mask_size: int, minimum size of the masks to keep
    :return: None, saves the split tiles into `tiles_dir/save_dir`
    """
    names = os.listdir(os.path.join(tiles_dir, sub_dir))
    save_dir = os.path.join(tiles_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    for name in names:
        if not name.endswith('.npz'): continue
        t = open_tile_npz(os.path.join(tiles_dir, sub_dir, name))
        tiles = split_tile(t)
        tiles = [remove_small_masks(t, min_mask_size) for t in tiles]
        tiles = [normalize_ndsm(t) if normalize else t for t in tiles]
        tiles = [normalize_orthophoto(t) if normalize else t for t in tiles]
        save_split_tiles(save_dir, name, tiles)


def select_tiles(tiles_dir: str, sub_dir: str = 'dataset', min_instances: int = 0, max_instances: int = 100) -> list[str]:
    """
    Return the name of the tiles that have between min_instances and max_instances instances.

    :param tiles_dir: str, path to the directory containing the tiles
    :param sub_dir: str, sub-directory where the tiles are saved
    :param min_instances: int, minimum number of instances
    :param max_instances: int, maximum number of instances
    :return: list[str], names of the selected tiles
    """
    tile_names = os.listdir(os.path.join(tiles_dir, sub_dir))
    selected_tiles = []
    for tile_name in tile_names:
        if not tile_name.endswith('.npz'): continue
        t = open_tile_npz(os.path.join(tiles_dir, sub_dir, tile_name))
        n_instances = np.max(t.instance_labels)
        if n_instances >= min_instances and n_instances <= max_instances:
            selected_tiles.append(tile_name)
    return selected_tiles


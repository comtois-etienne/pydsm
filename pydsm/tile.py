from dataclasses import dataclass
import osgeo
from osgeo import gdal, ogr, osr
import numpy as np
import pandas as pd
import os


import pydsm.nda as nda
import pydsm.geo as geo
import pydsm.utils as utils


@dataclass
class Tile:
    orthophoto: np.ndarray
    ndsm: np.ndarray
    instance_labels: np.ndarray
    semantic_labels: np.ndarray


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
        'SY': 18,
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


def get_semantic_code(semantics_dict: dict, code: str) -> int:
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
        int_code = get_semantic_code(semantics_dict, code)
        row_indexer = df['code'] == code
        df.loc[row_indexer, 'int_code'] = int_code

    v_list = df['v'].tolist()
    s_list = df['int_code'].tolist()

    return nda.replace_value_inplace(instance_labels, v_list, s_list)


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


def preprocess_tile(tile: Tile, ndsm_clip_height=30.0) -> Tile:
    """
    Preprocess a tile so it can be used for training a model.  

    :param tile: tile containing orthophoto, ndsm, instance labels, and semantic labels
    :param dsm_clip_height: float, height to which the DSM will be clipped.
    :return: Tile containing the processed `{orthophoto, dsm, labels, labels_downsampled, species, centers, centers_downsampled}`.  
        - `orthophoto` is normalized to `[0.0, 1.0]` range  
        - `ndsm` is clipped and rescaled to `[0.0, 1.0]` range where `1.0` equals `dsm_clip_height`
        - `instance_labels` are relabeled to have unique values starting from `0` (does not use skimage's label function)  
        - `semantic_labels` numpy array with the tree species codes `(int)` applied to the labels  
    """
    orthophoto = nda.normalize(tile.orthophoto[..., :3])
    ndsm = tile.ndsm[..., :1] if tile.ndsm.ndim > 2 else tile.ndsm
    ndsm = nda.clip_rescale(ndsm, ndsm_clip_height)
    instance_labels = nda.relabel(tile.instance_labels)

    return Tile(orthophoto, ndsm, instance_labels, tile.semantic_labels)


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


def create_tile_dataset(tiles_dir: str, save_sub_dir: str = 'dataset', semantic_dict=default_semantic_dict()) -> None:
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
    
    for name in names:
        if not name.endswith('.tif'): continue
        tile = open_tile(tiles_dir, name, semantic_dict)
        tile = preprocess_tile(tile)
        tiles = split_tile(tile)
        save_split_tiles(save_dir, name, tiles)


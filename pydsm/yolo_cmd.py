"""
Commands for processing geotiff files
"""


import os
from glob import glob
from typing import Any


import pydsm.tile as tile
import pydsm.utils as utils
import pydsm.yolo as yolo
from pydsm import const


from pydsm.cmd import _print


def predict_tile(
    models: str | list[str] | Any | list[Any],
    tiles_dir: str,
    tile_name: str,
    pred_mode: str = 'rgbd',
    model_input_size: float = 40,
    model_input_ndsm_max_value: float = const.CLIP_HEIGHT,
    verbose: bool = False,
) -> str:
    """
    Predict and save one tile from an organized tiles directory.

    :param models: YOLO model path, paths, loaded model, or loaded models.
    :param tiles_dir: directory containing orthophoto and nDSM subdirectories.
    :param tile_name: tile filename with or without extension.
    :param pred_mode: prediction mode accepted by yolo_rgbd.
    :param model_input_size: physical side length expected by the model.
    :param model_input_ndsm_max_value: nDSM height clipped before prediction.
    :param verbose: whether to print the processed tile.
    :return: path to the saved prediction.
    """
    tile_name = utils.remove_extension(tile_name)
    if verbose:
        _print(f'Processing prediction for tile: {tile_name}')
    result = yolo.predict_tile(
        models,
        os.path.join(tiles_dir, const.ORTHOPHOTO_SUBDIR, f'{tile_name}.tif'),
        os.path.join(tiles_dir, const.NDSM_SUBDIR, f'{tile_name}.tif'),
        pred_mode,
        model_input_size,
        model_input_ndsm_max_value,
    )
    prediction_dir = os.path.join(tiles_dir, const.PREDICTED_TILES_SUBDIR)
    os.makedirs(prediction_dir, exist_ok=True)
    prediction_path = os.path.join(prediction_dir, f'{tile_name}.npz')
    tile.save_tile(prediction_path, result)
    return prediction_path



def predict_tiles(
    models: str | list[str] | Any | list[Any],
    tiles_dir: str,
    pred_mode: str = 'rgbd',
    model_input_size: float = 40,
    model_input_ndsm_max_value: float = const.CLIP_HEIGHT,
    verbose: bool = False,
    skip_existing: bool = False,
) -> list[str]:
    """
    Predict and save every orthophoto tile in an organized directory.

    :param models: YOLO model path, paths, loaded model, or loaded models.
    :param tiles_dir: directory containing orthophoto and nDSM subdirectories.
    :param pred_mode: prediction mode accepted by yolo_rgbd.
    :param model_input_size: physical side length expected by the model.
    :param model_input_ndsm_max_value: nDSM height clipped before prediction.
    :param verbose: whether to print the processed tiles.
    :param skip_existing: whether to skip already saved predictions.
    :return: paths to the saved predictions.
    """
    orthophoto_dir = os.path.join(tiles_dir, const.ORTHOPHOTO_SUBDIR)
    models = yolo.load_models(models)
    tile_names = sorted(
        utils.remove_extension(os.path.basename(path))
        for path in glob(os.path.join(orthophoto_dir, '*.tif'))
    )
    prediction_dir = os.path.join(tiles_dir, const.PREDICTED_TILES_SUBDIR)
    paths: list[str] = []
    for tile_name in tile_names:
        prediction_path = os.path.join(prediction_dir, f'{tile_name}.npz')
        if skip_existing and os.path.isfile(prediction_path):
            if verbose:
                _print(f'Skipping existing prediction for tile: {tile_name}')
            paths.append(prediction_path)
            continue
        paths.append(predict_tile(
            models,
            tiles_dir,
            tile_name,
            pred_mode,
            model_input_size,
            model_input_ndsm_max_value,
            verbose,
        ))
    return paths



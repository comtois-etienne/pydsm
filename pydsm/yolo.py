import os
from math import log2
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import label

import pydsm.const as const
import pydsm.geo as geo
import pydsm.nda as nda
import pydsm.tile as tile
import pydsm.utils as utils

from yolo_rgbd.prediction import ObjectDetector as RGBDObjectDetector
from yolo_rgbd.prediction import predict_from_array

from .rbc import get_contour as rbc_get_contour

########### Anonymisation ###########
def __gaussian_blur_from_boxes(array: np.ndarray, boxes: pd.DataFrame, sigma: float = 5.0) -> np.ndarray:
    """
    :param ndarray: 3D array of the image
    :param boxes: DataFrame with the bounding boxes (xmin, ymin, xmax, ymax)
    :param sigma: sigma of the Gaussian filter
    :return: 3D array of the image with blurred bounding boxes
    """
    from skimage.filters import gaussian

    array = nda.normalize(array)
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


def anonymise_with_yolo(array: np.ndarray, model_name='yolov8n.pt') -> np.ndarray:
    """
    Blur the bounding boxes humans in the image using a Gaussian filter.
    
    :param ndarray: 3D array of the image
    :return: 3D array of the image with blurred bounding boxes
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    detector = RGBDObjectDetector(model)
    size = (max(array.shape) + 32) // 32 * 32
    detector.detect(array, imgsz=size)
    boxes = detector.get_objs_by_class(0, 0.2) # class 0 is person, confidence threshold 0.2
    return __gaussian_blur_from_boxes(array, boxes)


########### Segmentation Training Dataset ###########


@dataclass
class IndexLine:
    class_index: int # int, representing the predetermined object class index
    contour: np.ndarray # [[y1, x1], [y2, x2], ..., [yn, xn]]
    size: int # size of the image (assuming square image)

    def to_line(self) -> str:
        """
        convert a contour to a yolo class index string
        example: '1 0.5 0.5 0.1 0.1 0.3 0.3' (<class-index> <x1> <y1> <x2> <y2> ... <xn>)

        :param size: int, size of the image (assuming square image)
        :return: str, of the instance in yolo format
        """
        size = float(self.size - 1)
        contour = self.contour[:, ::-1]  # switch to (x, y)
        contour = contour.flatten().tolist()
        contour = [c / size for c in contour]
        contour = [f'{c:.3f}' for c in contour]
        return ' '.join([str(self.class_index)] + contour)
    
    def remap(self, remap_dict: dict) -> 'IndexLine':
        """
        remap the class_index using remap_dict

        :param remap_dict: dict, mapping old class_index to new class_index (-1 to remove)
        :return: IndexLine, with remapped class_index
        """
        new_class_index = remap_dict[self.class_index]
        if new_class_index == -1:
            return None
        return IndexLine(new_class_index, self.contour, self.size)


def to_index_lines(instance_labels: np.ndarray, semantic_labels: np.ndarray | None, *, edges=32, visualize=False) -> list[IndexLine]:
    """
    Convert mask instances and their class labels to yolo class index strings
    Each instance is represented by a convex hull contour of 8 sides
    The class index in `class_labels` must start at 1 (0 is background)
    The `class_index` values in `class_labels` are decremented by 1 to match yolo classes with 0 indexing
    Reference : https://docs.ultralytics.com/datasets/segment/

    :param instance_labels: np.ndarray of shape (H, W) with integer instance labels (0 for background)
    :param class_labels: np.ndarray of shape (H, W) with integer class labels (0 for background) or None if all instances are of the same class
    :param edges: int, number of edges for the contour approximation of each instance
    :param visualize: bool, if True, display the instances and their contours
    :return: list of str, each str representing one instance in yolo format 
    """
    unique = sorted(np.unique(instance_labels).tolist())
    unique = unique[1:] if unique[0] == 0 else unique
    instance_labels = instance_labels.copy()

    if semantic_labels is None:
        semantic_labels = (instance_labels > 0).astype(np.uint8)
    semantic_labels = semantic_labels.copy()
    size = instance_labels.shape[0]
    instance_labels = np.pad(instance_labels, ((100, 100), (100, 100)), mode='constant', constant_values=0)

    index_lines = []
    contours = []
    for val in unique:
        instance = np.where(instance_labels == val, 1, 0).astype(np.uint8)
        if np.sum(instance) < 100: continue

        is_split = len(np.unique(label(instance)).tolist()) > 2
        contour = rbc_get_contour(instance, edges=edges, convex_hull=is_split, drop_last=True) - np.array([100, 100])

        instance = instance[100:-100, 100:-100]
        class_index = np.max(semantic_labels * instance)

        contour[contour >= size] = size - 1
        index_line = IndexLine((class_index - 1), contour, size)
        index_lines.append(index_line)
        contours.append(contour) # for visualization

    if visualize:
        plt.imshow(instance_labels[100:-100, 100:-100], cmap='tab20', interpolation='nearest')
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color='red')
        plt.show()

    return index_lines


def save_indexlines(dir_npz: str, dir_labels: str, edges_per_instance=32, remap_dict=None, instance=False, visualize=False):
    """
    Convert all .npz tile files in `dir_npz` to .txt label files in YOLO format in `dir_labels`
    dataset structure:
    tiles/dataset/  
        train/  
            images/ -> npz tiles  

            labels/ -> txt labels  
        /val  
            images/  

            labels/  
    
    :param dir_npz: str, path to the directory containing the .npz tile files
    :param dir_labels: str, path to the directory where the .txt label files will be saved
    :param edges_per_instance: int, number of edges for the contour approximation of each instance
    :param ignore_class: list of int, class indices to ignore (not saved in the .txt files)
    :param instance: bool, if True, use instance labels only (all instances have the same class)
    :param visualize: bool, if True, display the tile name being processed
    :return: None, saves .txt files in YOLO format in dir_labels
    """
    os.makedirs(dir_labels, exist_ok=True)
    remap_dict = None if instance else remap_dict # ignore remap_dict if instance only

    for filename in os.listdir(dir_npz):
        if not filename.endswith('.npz'):
            continue

        tile_path = os.path.join(dir_npz, filename)
        t = tile.open_tile_npz(tile_path)

        if visualize: print(f'Processing \'{filename}\'')

        index_lines = to_index_lines(
            t.instance_labels, 
            None if instance else t.semantic_labels, 
            edges=edges_per_instance, 
            visualize=visualize
        )

        if remap_dict is not None:
            index_lines_remapped = []
            for line in index_lines:
                remapped_line = line.remap(remap_dict)
                index_lines_remapped.append(remapped_line) if remapped_line is not None else None
            index_lines = index_lines_remapped

        label_filename = filename.replace('.npz', '.txt')
        label_path = os.path.join(dir_labels, label_filename)

        index_lines_str = [line.to_line() for line in index_lines]
        with open(label_path, 'w') as f:
            f.write('\n'.join(index_lines_str))
            f.write('\n')


def save_one_indexlines(dir_npz: str, dir_labels: str, class_index=1, class_count=100, edges_per_instance=32, visualize=False):
    """
    Export only one class (other: 0 - selected_class: 1) to yolo format file.  
    See `save_to_class_index_file`  
    
    :param dir_npz: str, path to the directory containing the .npz tile files
    :param dir_labels: str, path to the directory where the .txt label files will be saved
    :param class_index: index of the class to export (0 is BG)
    :param class_count: the number of classes (excluding BG)
    :param edges_per_instance: int, number of edges for the contour approximation of each instance
    :param visualize: bool, if True, display the tile name being processed
    :return: None, saves .txt files in YOLO format in dir_labels
    """
    source_class = np.array(list(range(class_count)))
    target_class = np.array([0] * len(source_class))
    target_class[(class_index - 1)] = 1
    remap_dict = dict(zip(source_class, target_class))
    save_indexlines(dir_npz, dir_labels, edges_per_instance, remap_dict, visualize=visualize)


########### Segmentation Prediction ###########


def load_rgbd(orthophoto_path: str, ndsm_path: str, clip_height : float = const.CLIP_HEIGHT) -> np.ndarray:
    """
    Load and normalize the rgbd image from an orthophoto (geoTIFF) and ndsm (geoTIFF)  
    The rgb part is normalized and converted to uint8  
    the d part has its values above the `clip_height` clipped and then scaled to uint8
    - where `0` equals to 0.0 meters 
    - and `255` equals to 'clip_height' in meters
    Refer to the model specifications for the clip_height

    :param orthophoto_path: str, path to the orthophoto (geoTIFF)
    :param ndsm_path: str, path to the ndsm (geoTIFF)
    :param clip_height: float, ceiling of the ndsm on which the model has been trained on
    :return: np.ndarray, rgbd array with dtype uint8 for the YOLO model prediction
    """
    rgb = geo.to_ndarray(geo.open_geotiff(orthophoto_path))[..., :3]
    rgb = nda.to_uint8(rgb, norm=True)

    d = geo.to_ndarray(geo.open_geotiff(ndsm_path))
    d = nda.clip_rescale(d, clip_height)
    d = nda.to_uint8(d, norm=False)

    return np.dstack((rgb, d))


def _split_count(size: float, model_input_size: float) -> int:
    """Return the number of four-way splits required for a tile size."""
    if size < model_input_size or size % model_input_size:
        raise ValueError('Tile size must be a multiple of model_input_size')
    ratio = size / model_input_size
    count = int(log2(ratio))
    if 2 ** count != ratio:
        raise ValueError('Tile size must be model_input_size multiplied by 2^n')
    return count


def _combine_prediction_arrays(
    arrays: list[np.ndarray],
    combine_instances: bool,
) -> np.ndarray:
    """Combine four prediction arrays while preserving quadrant order."""
    if combine_instances:
        return nda.combine_four_instances(arrays)
    return np.block([[arrays[0], arrays[1]], [arrays[2], arrays[3]]])


def predict_tile(
    model_paths: str | list[str],
    orthophoto_path: str,
    ndsm_path: str,
    pred_mode: str = 'rgbd',
    model_input_size: float = 40,
    model_input_ndsm_max_value: float = 25.5,
) -> tile.Tile:
    """
    Predict a tile by splitting it into model-sized quadrants.

    :param model_paths: YOLO model path or paths.
    :param orthophoto_path: path to the orthophoto GeoTIFF.
    :param ndsm_path: path to the nDSM GeoTIFF.
    :param pred_mode: prediction mode accepted by yolo_rgbd.
    :param model_input_size: physical side length expected by the model.
    :param model_input_ndsm_max_value: nDSM height clipped before prediction.
    :return: tile containing the recombined predictions.
    """
    image = load_rgbd(
        orthophoto_path,
        ndsm_path,
        clip_height=model_input_ndsm_max_value,
    )
    size = geo.get_size(geo.open_geotiff(orthophoto_path))[0]
    split_count = _split_count(size, model_input_size)
    images = [image]
    for _ in range(split_count):
        images = [part for current in images for part in nda.split_four(current)]

    prediction_images = [part[:, :, :3] if pred_mode == 'rgb' else part for part in images]
    predictions = [
        predict_from_array(
            model_paths,
            part,
            pred_mode=pred_mode,
            confidence=const.CONFIDENCE_THRESHOLD,
        )
        for part in prediction_images
    ]
    for _ in range(split_count):
        predictions = [
            tuple(
                _combine_prediction_arrays(
                    [prediction[index] for prediction in predictions[offset:offset + 4]],
                    index == 0,
                )
                for index in range(3)
            )
            for offset in range(0, len(predictions), 4)
        ]
    instances, semantics, confidences = predictions[0]
    result = tile.Tile(image[:, :, :3], image[:, :, 3], instances, semantics)
    result.confidence_labels = confidences
    return result


def predict_from_geotiff(model_paths: str | list[str], tiles_dir: str, tile_name: str, pred_mode='rgb', verbose=False) -> None:
    """
    Predict the instance segmentation masks in one tile using a YOLO model.

    :param model_path: str, path to the YOLO model trained on RGB-D images
    :param tiles_dir: str, path to the directory containing the tiles
    :param tile_name: str, name of the tile (with or without extension)
    :param pred_mode: str, 'rgb' or 'rgbd', whether to use only RGB channels or RGB-D channels for prediction
    :param verbose: bool, whether to display the prediction results
    :return: None, saves the prediction results as Tile (numpy array) on disk in const.PREDICTION_SUBDIR
    """
    model_paths = [model_paths] if isinstance(model_paths, str) else model_paths
    tile_name = utils.remove_extension(tile_name)

    t = predict_tile(
        model_paths,
        os.path.join(tiles_dir, const.ORTHOPHOTO_SUBDIR, f'{tile_name}.tif'),
        os.path.join(tiles_dir, const.NDSM_SUBDIR, f'{tile_name}.tif'),
        pred_mode=pred_mode,
    )

    if verbose: 
        tile.display_tile(t)
        plt.imshow(t.confidence_labels, vmin=0, vmax=1, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Confidence', orientation='vertical')
        plt.show()

    return t


def predict_from_geotiffs(model_paths: str | list[str], tiles_dir: str, pred_mode='rgb', verbose=False, skip_existing=False) -> None:
    """
    Predict the instance segmentation masks in multiple tiles using the combined prediction from multiple YOLO models.

    :param model_paths: list[str], path to the YOLO models trained on RGB-D or RGB images
    :param tiles_dir: str, path to the directory containing the tiles
    :param tile_names: list[str], names of the tiles (with or without extension)
    :param pred_mode: str, 'rgb' or 'rgbd', whether to use only RGB channels or RGB-D channels for prediction
    :param verbose: bool, whether to display the prediction results
    :param skip_existing: bool, whether to skip tiles that already have prediction results saved
    :return: None, saves the prediction results as Tile (numpy array) on disk in const.PREDICTION_SUBDIR
    """
    tile_names = [utils.remove_extension(f) for f in os.listdir(os.path.join(tiles_dir, const.ORTHOPHOTO_SUBDIR)) if f.endswith('.tif')]
    tile_names = sorted(tile_names)
    for tile_name in tile_names:
        t_path = os.path.join(tiles_dir, const.PREDICTED_TILES_SUBDIR, f'{tile_name}.npz')
        if skip_existing:
            if os.path.exists(t_path):
                if verbose: print(f'Skipping existing prediction for tile \'{tile_name}\'')
                continue
        t = predict_from_geotiff(model_paths, tiles_dir, tile_name, pred_mode, verbose=verbose)
        tile.save_tile(t_path, t)

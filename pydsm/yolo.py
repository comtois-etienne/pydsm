from dataclasses import dataclass
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.morphology import opening as binary_opening
import os

from skimage.measure import label

from pathlib import Path
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

from .rbh import get_contour as rbh_get_contour
import pydsm.utils as utils
import pydsm.geo as geo
import pydsm.nda as nda
import pydsm.tile as tile
import pydsm.const as const


########### RGBD Monkey Patching ###########
"""
Optional, for viewing images during training
in ultralytics.utils.plotting.py
change original line in function `plot_images` :
`mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0) # Original`

to this line :
`mosaic[y : y + h, x : x + w, :] = images[i][[3,2,1]].transpose(1, 2, 0) # Modified`
"""

from ultralytics.data.base import IMG_FORMATS
from ultralytics import YOLO # ensure YOLO is imported before monkey patching
from PIL import Image
import cv2


if cv2.__dict__.get('original_imread', None) is None:
    print("Monkey Patching 🐒 cv2.imread to add .npz tiles support for YOLO RGBD")
    cv2.original_imread = cv2.imread

if Image.__dict__.get('original_open', None) is None:
    print("Monkey Patching 🐒 PIL.Image.open to add .npz tiles support for YOLO RGBD")
    Image.original_open = Image.open


def fake_cv2_imread(file_path):
    # print('Called fake_imread')
    if utils.get_extension(file_path) == 'npz':
        t = tile.open_tile_npz(file_path)
        return t.rgbd()
    else:
        return cv2.original_imread(file_path)


def fake_pil_open(file_path):
    # print('Called fake_open')
    if utils.get_extension(file_path) == 'npz':
        t = tile.open_tile_npz(file_path)
        rgbd = t.rgbd()
        pil_im = Image.fromarray(rgbd, mode='RGBA')
        pil_im.format = 'NPZ'
        return pil_im
    else:
        return Image.original_open(file_path)


IMG_FORMATS.add("npz")
cv2.imread = fake_cv2_imread
Image.open = fake_pil_open


########### Anonymisation ###########


coco_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


def _get_objs(res):
    columns=['xmin', 'ymin', 'xmax', 'ymax', 'xcenter', 'score', 'class', 'name']
    df = pd.DataFrame(columns=columns)
    for r in res:
        for i in range(len(r.boxes.cls)):
            xyxy = np.array(r.boxes.xyxy[i].cpu(), dtype=int)
            df = pd.concat([df, pd.DataFrame({
                    'xmin': xyxy[0],
                    'ymin': xyxy[1],
                    'xmax': xyxy[2],
                    'ymax': xyxy[3],
                    'xcenter': int((xyxy[0] + xyxy[2]) / 2),
                    'score': float(r.boxes.conf[i]),
                    'class': int(r.boxes.cls[i]),
                    'name': coco_classes[int(r.boxes.cls[i])],
                }, index=[0])],
            ) 
    return df


def plot_obj(ax, obj, color='r', linewidth=2):
    rect = patches.Rectangle((obj['xmin'], obj['ymin']), obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin'], linewidth=linewidth, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def plot_objs(ax, objs, color='r', linewidth=2):
    for i in range(len(objs)):
        plot_obj(ax, objs.iloc[i], color=color, linewidth=linewidth)


class ObjectDetector:

    def __init__(self, model):
        self.model = model
        self.results = None

    def detect(self, image, verbose=False, imgsz=640):
        self.results = self.model(image, verbose=verbose, imgsz=imgsz)
        return self.results

    def get_objs(self, score=0.5):
        objs = _get_objs(self.results)
        return objs[objs['score'] >= score]

    def get_objs_by_name(self, cls, score=0.5):
        objs = self.get_objs(score)
        return objs[objs['name'] == cls]

    def get_objs_by_class(self, cls, score=0.5):
        objs = self.get_objs(score)
        return objs[objs['class'] == cls]


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
    from pydsm.yolo import ObjectDetector

    model = YOLO(model_name)
    detector = ObjectDetector(model)
    size = (max(array.shape) + 32) // 32 * 32
    detector.detect(array, imgsz=size)
    boxes = detector.get_objs_by_name('person', 0.2)
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
        contour = rbh_get_contour(instance, edges=edges, convex_hull=is_split, drop_last=True) - np.array([100, 100])

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


def predict_instances(model_path: str, image: np.ndarray, confidence=const.CONFIDENCE_THRESHOLD) -> tuple[list[np.array], list[float], list[int]]:
    """
    Predict the instance segmentation masks in one image using a YOLO model.

    :param model_path: str, path to the YOLO model trained on RGB-D or RGB images
    :param image: np.ndarray (dtype=unint8), input RGB-D or RGB image as a numpy array of shape (H, W, 4) (RGBD) or (H, W, 3) (RGB)
    :param confidence: float, confidence threshold for predictions
    :return: tuple of lists containing:
        - list of np.ndarray, predicted instance segmentation masks of shape (H, W) with binary values (0 or 1)
        - list of float, confidence scores for each predicted mask
        - list of int, class labels for each predicted mask
    """
    model = YOLO(model_path)
    results = model.predict(source=image, conf=confidence, verbose=False)

    orig_shape = image.shape[:2]
    if results[0].masks is None:
        return [np.zeros(orig_shape, dtype=np.uint16)], [0.0], [0]

    masks = results[0].masks.data.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = [int(c) for c in results[0].boxes.cls.cpu().numpy()]

    pred_shape = masks[0].shape[:2]
    if pred_shape != orig_shape:
        masks = [nda.rescale_nearest_neighbour(mask, orig_shape) for mask in masks]

    return masks, confs, classes


def clean_prediction(masks: list[np.ndarray], confs: list[float], classes: list[int]) -> tuple[np.ndarray]:
    """
    Merges the predicted instance segmentation masks into a single instance mask and a semantic mask. 
    Removes holes, applys non-maximum suppression (NMS), and assigs instance and semantic labels.

    :param masks: list of np.ndarray, predicted instance segmentation masks of shape (H, W) with binary values (0 or 1)
    :param confs: list of float, confidence scores for each predicted mask
    :param classes: list of int, class labels for each predicted mask
    :return: tuple containing:
        - np.ndarray, instance mask of shape (H, W) with integer values representing instance labels (0 for background, 1 for first instance, 2 for second instance, etc.)
        - np.ndarray, semantic mask of shape (H, W) with integer values representing class labels (0 for background, 1 for first class, 2 for second class, etc.)
        - np.ndarray, confidence mask of shape (H, W) with float values representing confidence scores for each pixel
    """
    masks = [nda.get_biggest_mask(mask) for mask in masks]
    masks = [nda.remove_holes(mask) for mask in masks]
    kept_masks = nda.nms(masks, confs, const.IOU_THRESHOLD)[::-1]
    kept_masks_index = [np.where(np.all(masks == m, axis=(1, 2)))[0][0] for m in kept_masks]

    out_shape = masks[0].shape[:2]
    instances = np.zeros(out_shape, dtype=np.uint16)
    semantics = np.zeros(out_shape, dtype=np.uint16)
    confidences = np.zeros(out_shape, dtype=np.float32)

    for i, mask in enumerate(kept_masks):
        instances[mask > 0] = i + 1

    new_instances = np.zeros_like(instances)
    instances = nda.remove_small_masks(instances, min_area=const.MIN_MASK_SIZE)
    for i in np.unique(instances):
        if i == 0: continue
        mask = instances == i
        mask = binary_opening(mask, disk(const.REMOVE_CRACKS_SIZE))
        mask = nda.get_biggest_mask(mask)
        new_instances[mask > 0] = i
    instances = new_instances

    kept_masks = [instances == (i + 1) for i in range(instances.max())] # clears the removed small masks

    for i, mask in enumerate(kept_masks):
        index = kept_masks_index[i]
        semantics[mask > 0] = classes[index] + 1
        confidences[mask > 0] = confs[index]

    instances = nda.relabel(instances)
    return instances, semantics, confidences


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
    thresh = const.CONFIDENCE_THRESHOLD
    model_paths = [model_paths] if isinstance(model_paths, str) else model_paths
    tile_name = utils.remove_extension(tile_name)

    image = load_rgbd(
        os.path.join(tiles_dir, const.ORTHOPHOTO_SUBDIR, f'{tile_name}.tif'),
        os.path.join(tiles_dir, const.NDSM_SUBDIR, f'{tile_name}.tif')
    )
    rgb, d = image[:, :, :3], image[:, :, 3]
    image = rgb if pred_mode == 'rgb' else image

    masks, confs, classes = [], [], []
    for model_path in model_paths:
        ms, cs, cls = predict_instances(model_path, image, thresh)
        masks.extend(ms)
        classes.extend(cls)
        confs.extend([c * thresh if cl == 0 else c for c, cl in zip(cs, cls)])

    instances, semantics, confidences = clean_prediction(masks, confs, classes)
    confidences[confidences < thresh] *= (1/thresh)

    t = tile.Tile(rgb, d, instances, semantics)
    t.confidence_labels = confidences
    if verbose: 
        tile.display_tile(t)
        plt.imshow(t.confidence_labels, vmin=0, vmax=1, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Confidence', orientation='vertical')
        plt.show()

    return t


def predict_from_geotiffs(model_paths: str | list[str], tiles_dir: str, pred_mode='rgb', verbose=False) -> None:
    """
    Predict the instance segmentation masks in multiple tiles using the combined prediction from multiple YOLO models.

    :param model_paths: list[str], path to the YOLO models trained on RGB-D or RGB images
    :param tiles_dir: str, path to the directory containing the tiles
    :param tile_names: list[str], names of the tiles (with or without extension)
    :param pred_mode: str, 'rgb' or 'rgbd', whether to use only RGB channels or RGB-D channels for prediction
    :param verbose: bool, whether to display the prediction results
    :return: None, saves the prediction results as Tile (numpy array) on disk in const.PREDICTION_SUBDIR
    """
    tile_names = [utils.remove_extension(f) for f in os.listdir(os.path.join(tiles_dir, const.ORTHOPHOTO_SUBDIR)) if f.endswith('.tif')]
    tile_names = sorted(tile_names)
    for tile_name in tile_names:
        t = predict_from_geotiff(model_paths, tiles_dir, tile_name, pred_mode, verbose=verbose)
        t_path = os.path.join(tiles_dir, const.PREDICTED_TILES_SUBDIR, f'{tile_name}.npz')
        tile.save_tile(t_path, t)


import matplotlib.patches as patches
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os

from skimage.measure import label

from pathlib import Path
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

from .rbh import get_contour as rbh_get_contour
from .nda import to_uint8 as nda_to_uint8
from .tile import open_tile_npz as tile_open_tile_npz
from .tile import Tile


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


def to_class_index_line(contour: np.ndarray, class_index: int, size: int) -> str:
    """
    convert a contour to a yolo class index string
    example: '1 0.5 0.5 0.1 0.1 0.3 0.3' (<class-index> <x1> <y1> <x2> <y2> ... <xn>)

    :param contour: np.ndarray [[y1, x1], [y2, x2], ..., [yn, xn]] of one instance
    :param class_index: int, representing the predetermined object class index
    :param size: int, size of the image (assuming square image)
    :return: str, of the instance in yolo format
    """
    size = float(size - 1)
    contour = contour[:, ::-1]  # switch to (x, y)
    contour = contour.flatten().tolist()
    contour = [c / size for c in contour]
    contour = [f'{c:.3f}' for c in contour]
    return ' '.join([str(class_index)] + contour)


def to_class_index_lines(instance_labels: np.ndarray, semantic_labels: np.ndarray | None, *, edges=8, visualize=False) -> list[str]:
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

    class_index_lines = []
    contours = []
    for val in unique:
        instance = np.where(instance_labels == val, 1, 0).astype(np.uint8)
        if np.sum(instance) < 100: continue

        is_split = len(np.unique(label(instance)).tolist()) > 2
        contour = rbh_get_contour(instance, edges=edges, convex_hull=is_split, drop_last=True) - np.array([100, 100])

        instance = instance[100:-100, 100:-100]
        class_index = np.max(semantic_labels * instance)

        contour[contour >= size] = size - 1
        class_index_line = to_class_index_line(contour, (class_index - 1), size)
        class_index_lines.append(class_index_line)
        contours.append(contour)

    if visualize:
        plt.imshow(instance_labels[100:-100, 100:-100], cmap='tab20', interpolation='nearest')
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color='red')
        plt.show()

    return class_index_lines


def save_to_class_index_file(dir_npz: str, dir_labels: str, edges_per_instance=8, visualize=False):
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
    :param visualize: bool, if True, display the tile name being processed
    :return: None, saves .txt files in YOLO format in dir_labels
    """
    os.makedirs(dir_labels, exist_ok=True)

    for filename in os.listdir(dir_npz):
        if not filename.endswith('.npz'):
            continue

        tile_path = os.path.join(dir_npz, filename)
        tile = tile_open_tile_npz(tile_path)

        if visualize: print(f'Processing \'{filename}\'')

        class_index_lines = to_class_index_lines(
            tile.instance_labels, 
            tile.semantic_labels, 
            edges=edges_per_instance, 
            visualize=visualize
        )

        label_filename = filename.replace('.npz', '.txt')
        label_path = os.path.join(dir_labels, label_filename)

        with open(label_path, 'w') as f:
            f.write('\n'.join(class_index_lines))
            f.write('\n')


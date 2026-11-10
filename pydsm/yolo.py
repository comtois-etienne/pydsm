import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

import pydsm.nda as nda
import pydsm.tile as tile

from .rbc import get_contour as rbc_get_contour


########### Anonymisation ###########
def __gaussian_blur_from_boxes(
    array: np.ndarray,
    boxes: np.ndarray,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    :param ndarray: 3D array of the image
    :param boxes: array of bounding boxes (xmin, ymin, xmax, ymax)
    :param sigma: sigma of the Gaussian filter
    :return: 3D array of the image with blurred bounding boxes
    """
    from skimage.filters import gaussian

    array = nda.normalize(array)
    mask = np.zeros_like(array, dtype=np.float64)
    means = array.copy()
    for xmin, ymin, xmax, ymax in boxes.astype(int):
        mean = np.mean(array[ymin:ymax, xmin:xmax], axis=(0, 1))
        mask[ymin:ymax, xmin:xmax] = 1
        means[ymin:ymax, xmin:xmax] = mean

    blurred = gaussian(array, sigma=sigma)
    mask = gaussian(mask, sigma=sigma)

    new_array = array * (1 - mask) + blurred * mask
    new_array = (new_array + means) / 2

    return new_array


def _person_boxes(model: Any, array: np.ndarray, size: int) -> np.ndarray:
    """Return person bounding boxes from an Ultralytics prediction."""
    results = model.predict(
        array,
        imgsz=size,
        classes=[0],
        conf=0.2,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return np.empty((0, 4), dtype=int)
    return results[0].boxes.xyxy.cpu().numpy()


def anonymise_with_yolo(
    array: np.ndarray,
    model_name: str = 'yolov8n.pt',
) -> np.ndarray:
    """
    Blur the bounding boxes humans in the image using a Gaussian filter.
    
    :param ndarray: 3D array of the image
    :return: 3D array of the image with blurred bounding boxes
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    size = (max(array.shape) + 32) // 32 * 32
    boxes = _person_boxes(model, array, size)
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

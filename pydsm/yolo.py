import matplotlib.patches as patches
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os

from pathlib import Path
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

from .rbh import get_contour as rbh_get_contour
from .nda import to_uint8 as nda_to_uint8


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


def to_class_index_lines(instance_labels: np.ndarray, class_labels: np.ndarray, visualize=False) -> list[str]:
    """
    Convert mask instances and their class labels to yolo class index strings
    Each instance is represented by a convex hull contour of 8 sides
    The class index in `class_labels` must start at 1 (0 is background)
    The `class_index` values in `class_labels` are decremented by 1 to match yolo classes with 0 indexing

    :param instance_labels: np.ndarray of shape (H, W) with integer instance labels (0 for background)
    :param class_labels: np.ndarray of shape (H, W) with integer class labels (0 for background)
    :return: list of str, each str representing one instance in yolo format 
    """
    unique = sorted(np.unique(instance_labels).tolist())
    unique = unique[1:] if unique[0] == 0 else unique
    instance_labels = instance_labels.copy()
    class_labels = class_labels.copy()
    size = instance_labels.shape[0]
    instance_labels = np.pad(instance_labels, ((100, 100), (100, 100)), mode='constant', constant_values=0)

    class_index_lines = []
    for val in unique:
        instance = np.where(instance_labels == val, 1, 0).astype(np.uint8)
        contour = rbh_get_contour(instance, edges=8, convex_hull=True, drop_last=True) - np.array([100, 100])
        instance = instance[100:-100, 100:-100]
        class_index = np.max(class_labels * instance)

        contour[contour >= size] = size - 1
        class_index_line = to_class_index_line(contour, (class_index - 1), size)
        class_index_lines.append(class_index_line)

        if visualize:
            plt.imshow(instance)
            plt.plot(contour[:, 1], contour[:, 0], color='red')
            plt.show()

    return class_index_lines


def save_to_class_index_file(dir_npz: str, dir_labels: str, visualize=False):
    os.makedirs(dir_labels, exist_ok=True)

    for filename in os.listdir(dir_npz):
        if not filename.endswith('.npz'):
            continue

        tile_path = os.path.join(dir_npz, filename)
        tile_data = np.load(tile_path)

        if visualize: print(f'Processing \'{filename}\'')
        class_index_lines = to_class_index_lines(tile_data['labels'], tile_data['species'], visualize)
        label_filename = filename.replace('.npz', '.txt')
        label_path = os.path.join(dir_labels, label_filename)

        with open(label_path, 'w') as f:
            f.write('\n'.join(class_index_lines))
            f.write('\n')


class RGBDDataset(YOLODataset):
    """
    Override YOLODataset to load RGBD images from .npz files
    Each .npz file must contain : 'orthophoto' and 'dsm' arrays
    """
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im = self.ims[i] # cached in RAM
        f = self.im_files[i] # image path
        fn = self.npy_files[i] # cached npy path

        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                # im = cv2.imread(f)  # BGR (old)
                data = np.load(f)
                rgb = data['orthophoto'][:, :, :3]
                d = data['dsm']
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                im = nda_to_uint8(np.dstack((bgr, d)))
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]


"""
Dependencies:
- ultralytics
- numpy
- cv2
- Pillow
"""




########### Training Constants ###########

yolo_seg_n = 'yolo11n-seg.pt' # 2.9M
yolo_seg_s = 'yolo11s-seg.pt' # 10.1M
yolo_seg_m = 'yolo11m-seg.pt' # 22.4M
yolo_seg_l = 'yolo11l-seg.pt' # 27.6M
yolo_seg_x = 'yolo11x-seg.pt' # 62.1M


########### Training Variables ###########

MODEL = yolo_seg_n
MODE = 'depth'  # 'rgbd' or 'rgb'
YAML = f'/Users/etiennecomtois/Downloads/tiles/dataset_800/tree-instance-{MODE}.yaml'

EPOCHS = 5
IMGSZ = 128
BATCH_SIZE = 16
SAVE_PERIOD = 5




########### RGBD Monkey Patching ###########
"""
Optional, for viewing images during training
in ultralytics.utils.plotting.py
change original line in function `plot_images` :
`mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0) # Original`

to this line :
`mosaic[y : y + h, x : x + w, :] = images[i][[3,2,1]].transpose(1, 2, 0) # Modified`
"""

import numpy as np
from ultralytics.data.base import IMG_FORMATS
from ultralytics import YOLO # ensure YOLO is imported before monkey patching
from PIL import Image
import cv2


def open_tile_npz(npz_path: str, mode: str) -> np.ndarray:
    """
    Opens a tile that has been saved as a compressed numpy array.  
    The npz dtype must be uint8 and contain 'orthophoto' and 'ndsm' arrays.  

    :param npz_path: str, path to the tile
    :param mode: str, 'bgrd', 'rgbd', 'rgb', 'bgr', 'depth'
    :return: np.ndarray, the tile as a numpy array
    """
    npz = np.load(npz_path, allow_pickle=True)
    rgb = npz['orthophoto']
    d = npz['ndsm']

    if mode == 'rgbd':
        return np.dstack((rgb, d))
    if mode == 'rgb':
        return rgb
    if mode == 'bgrd':
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return np.dstack((bgr, d))
    if mode == 'bgr':
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if mode == 'depth':
        return np.dstack((d, d, d))


if cv2.__dict__.get('original_imread', None) is None:
    print("Monkey Patching 🐒 cv2.imread to add .npz tiles support for YOLO RGBD")
    cv2.original_imread = cv2.imread


if Image.__dict__.get('original_open', None) is None:
    print("Monkey Patching 🐒 PIL.Image.open to add .npz tiles support for YOLO RGBD")
    Image.original_open = Image.open


def fake_cv2_imread(file_path):
    """
    cv2.imread replacement to open .npz tiles as rgbd or rgb images.
    cv2 uses BGR color mode by default.
    """
    if file_path[-3:] == 'npz':
        return open_tile_npz(file_path, MODE)
    else:
        return cv2.original_imread(file_path)


def fake_pil_open(file_path):
    """
    PIL.Image.open replacement to open .npz tiles as rgbd or rgb images. 
    PIL uses RGB color mode by default.
    """
    if file_path[-3:] == 'npz':
        rgbd = open_tile_npz(file_path, MODE)
        pil_im = Image.fromarray(rgbd, mode=MODE)
        pil_im.format = 'NPZ'
        return pil_im
    else:
        return Image.original_open(file_path)


IMG_FORMATS.add("npz")
cv2.imread = fake_cv2_imread
Image.open = fake_pil_open




########### Training ###########

model = YOLO(YAML).load(MODEL)
results = model.train(data=YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH_SIZE, save_period=SAVE_PERIOD)
print(results.results_dict)
print(results.fitness)




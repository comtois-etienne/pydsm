#!/home/comet/projects/def-mjmeurs/comet/venv_ultralytics/bin/env python

"""
Dependencies:
- ultralytics
"""


import datetime
import sys
import os



########### Training Variables ###########

COLOR_MODE = sys.argv[1]
source_dataset = sys.argv[2]
training_dir = sys.argv[3]
print(f'\'{training_dir}\'')

# COLOR_MODE = 'rgbd'  # 'rgbd' or 'rgb' or 'depth'
# source_dataset = '/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_species/cv/fold_1'
# source_dataset = '/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_instance/cv/fold_1'
# training_dir = '/Users/etiennecomtois/Downloads/comet/100/yolo11l_species_rgbd_1984_f3_100e_(20260225 22h28)'


YAML = f'{source_dataset}/{COLOR_MODE}.yaml'
MODEL = f'{training_dir}/weights/best.pt'

log_path = f'{training_dir}/validation.log'
log_file = open(log_path, 'w')
sys.stdout = log_file


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
import scipy.stats


def _get_tile_mean_std() -> dict:
    """
    Based on 1000 tiles
    """
    return {
        'r_mean': 68.806,
        'r_std': 45.867,
        'g_mean': 91.519,
        'g_std': 42.803,
        'b_mean': 50.144,
        'b_std': 44.72,
        'd_mean': 4.800,
        'd_std': 5.030
    }


def _my_distribution(mean, std, min_val=0, max_val=255):
    # from `https://stackoverflow.com/questions/50626710/generating-random-numbers-with-predefined-mean-std-min-and-max`
    scale = max_val - min_val
    location = min_val
    unscaled_mean = (mean - min_val) / scale
    unscaled_var = (std / scale) ** 2
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
    alpha = beta * t
    return scipy.stats.beta(alpha, beta, scale=scale, loc=location)


def _random_layer(mean, std, size=128):
    dist = _my_distribution(mean, std)
    layer = dist.rvs(size=(size, size)).astype(np.uint8)
    layer = np.clip(layer, 0, 255)
    return layer


def random_tile(size=128):
    """
    :param size: int, size of the tile (size x size)
    """
    mean_std = _get_tile_mean_std()
    r = _random_layer(mean_std['r_mean'], mean_std['r_std'], size)
    g = _random_layer(mean_std['g_mean'], mean_std['g_std'], size)
    b = _random_layer(mean_std['b_mean'], mean_std['b_std'], size)
    d = _random_layer(mean_std['d_mean']*10, mean_std['d_std']*10, size)
    return np.dstack((r, g, b, d))


def convert(rgbd: np.ndarray, mode='rgbd'):
    """
    :param rgbd: np.ndarray, must be in the rgbd order
    :param mode: string, defines the order of the layers (rgb, bgr, ddd, bgrd...)
    :return: np.nparray, the reordered array
    """
    indexes = { 'r' : 0, 'g' : 1, 'b' : 2, 'd' : 3 }
    arrays = [rgbd[:,:,(indexes[m])] for m in list(mode)]
    return np.dstack(arrays)


def open_tile_npz(npz_path: str, mode: str) -> np.ndarray:
    """
    Opens a tile that has been saved as a compressed numpy array.  
    The npz dtype must be uint8 and contain 'orthophoto' and 'ndsm' arrays.  

    :param npz_path: str, path to the tile
    :param mode: str, 'bgrd', 'rgbd', 'rgb', 'bgr', 'depth', ...
    :return: np.ndarray, the tile as a numpy array
    """
    mode = 'ddd' if mode == 'depth' else mode
    npz = np.load(npz_path, allow_pickle=True)
    rgbd = np.dstack((npz['orthophoto'], npz['ndsm']))
    return convert(rgbd, mode)


if cv2.__dict__.get('original_imread', None) is None:
    print("Monkey Patching 🐒 cv2.imread to add .npz tiles support for YOLO RGBD")
    cv2.original_imread = cv2.imread


if Image.__dict__.get('original_open', None) is None:
    print("Monkey Patching 🐒 PIL.Image.open to add .npz tiles support for YOLO RGBD")
    Image.original_open = Image.open


def fake_cv2_imread(file_path):
    """
    This function open the tiles during the training process.  
    cv2.imread replacement to open .npz tiles as rgbd or rgb images.
    cv2 uses BGR color mode by default.
    """
    if file_path[-3:] == 'npz':
        tile = open_tile_npz(file_path, COLOR_MODE)
        # rand = np.random.rand()
        # if COLOR_MODE == 'rgbd' and rand < (DROP_DEPTH + DROP_RGB):
        #     fake_tile = random_tile(size=tile.shape[0])
        #     if rand < DROP_DEPTH: tile[:, :, 3] = fake_tile[:, :, 3]
        #     else: tile[:, :, 0:3] = fake_tile[:, :, 0:3]
        return tile
    else:
        return cv2.original_imread(file_path)


def fake_pil_open(file_path, *args, **kwargs):
    """
    This function open the tiles during the dataset integrity check.  
    PIL.Image.open replacement to open .npz tiles as rgbd or rgb images. 
    PIL uses RGB color mode by default.
    """
    if isinstance(file_path, str) and file_path.endswith(".npz"):
    # if file_path[-3:] == 'npz':
        rgbd = open_tile_npz(file_path, COLOR_MODE)
        mode = 'RGBA' if len(COLOR_MODE) == 4 else 'RGB'
        pil_im = Image.fromarray(rgbd, mode=mode)
        pil_im.format = 'NPZ'
        return pil_im
    else:
        return Image.original_open(file_path, *args, **kwargs)


IMG_FORMATS.add("npz")
cv2.imread = fake_cv2_imread
Image.open = fake_pil_open




########### Training ###########

print(f'\'{training_dir}\'')
print(f'{MODEL=}')
print(f'{YAML=}')


model = YOLO(MODEL)
results = model.val(
    data=YAML,
    project=training_dir,
    save=True,
    save_txt=True,
    save_conf=True,
    plots=True
)

print(results.results_dict)
print(results.fitness)

log_file.close()


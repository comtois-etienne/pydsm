import os
os.environ["SM_FRAMEWORK"] = "tf.keras" # Use TensorFlow Keras backend for segmentation_models
import osgeo
import numpy as np
import pandas as pd
import imageio.v3 as iio
from skimage.measure import label
from skimage.morphology import disk
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
from typing import Callable
from scipy.ndimage import median_filter

import pydsm.nda as nda
import pydsm.geo as geo
import pydsm.utils as utils

from keras.utils import to_categorical

from skimage.util import random_noise
from scipy import ndimage
from skimage import exposure


# SAVE NPZ TILES FUNCTIONS

def get_tree_species_dict() -> dict:
    """
    arrière-plan + inconnu + 5 especes + 10 genres + 1 famille
    18 classes total
    """
    UNKNOWN_TREE = 1
    BETULACEAE = 17
    return {
        'BACKGROUND': 0,
        'UNKNOWN_TREE': UNKNOWN_TREE,
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
        'BE': BETULACEAE,
        'OS': BETULACEAE,
        'AL': BETULACEAE,
        'CA': BETULACEAE,
    }


def get_tree_species_code(species: str) -> int:
    tree_species = get_tree_species_dict()
    species_long = species.upper()[:4]
    species_short = species_long[:2]

    if species_long in tree_species:
        return tree_species[species_long]
    
    if species_short in tree_species:
        return tree_species[species_short]

    return tree_species['UNKNOWN_TREE']


def apply_tree_species_codes(labels: np.ndarray, trees_df: pd.DataFrame, code_replacement_function: Callable = get_tree_species_code) -> np.ndarray:
    """
    :param array: numpy array, dtype must be integer. contains the mask instances of the trees
    :param trees_df: pandas DataFrame containing the tree position with their species (must contain ['axis-0', 'axis-1', 'code'])
    :param code_replacement_function: function that takes a species string and returns an integer species code
    :return: np.ndarray with the species codes applied to the array
    """
    unique_values = np.unique(labels)
    unique_values = unique_values[1:] if unique_values[0] == 0 else unique_values

    trees = trees_df[['axis-0', 'axis-1', 'code']].copy()
    trees.rename(columns={'axis-0': 'y', 'axis-1': 'x', 'code': 'species'}, inplace=True)

    trees['v'] = 0
    for i, (y, x) in enumerate(zip(trees['y'], trees['x'])):
        row_indexer = trees.index == i
        trees.loc[row_indexer, 'v'] = labels[int(y), int(x)]
    trees = trees[trees['v'] != 0]
    trees = trees.drop(columns=['x', 'y'])

    missing_species = []
    for v in unique_values:
        if v not in trees['v'].values:
            missing_species.append(v)

    for v in missing_species:
        df = pd.DataFrame({'species': '', 'v': v}, index=[0])
        trees = pd.concat([trees, df], ignore_index=True)

    trees['species_code'] = 0
    for species in trees['species']:
        code = code_replacement_function(species)
        row_indexer = trees['species'] == species
        trees.loc[row_indexer, 'species_code'] = code

    v_list = trees['v'].tolist()
    s_list = trees['species_code'].tolist()

    return nda.replace_value_inplace(labels, v_list, s_list)


def downsample_species_array(array_species: np.ndarray, size=100) -> np.ndarray:
    from scipy.ndimage import median_filter
    array_species_downsampled = nda.rescale_nearest_neighbour(array_species, (size, size))
    array_species_downsampled = median_filter(array_species_downsampled, size=3)
    return array_species_downsampled


def get_centers(labels: np.ndarray, downsampling_size=100):
    """
    Returns the centers of the instances in the labels array.

    :param labels: np.ndarray with shape (H, W) representing the instance segmentation labels.
    :param downsampling_size: int, size to which the labels will be downsampled.
    :return: tuple of two np.ndarrays:
        - array_centers: binary numpy array with the centers of the instances set to `1.0` (size of `labels`)
        - array_centers_downsampled: binary numpy array with the centers of the downsampled instances (size `downsampling_size`)
    """
    labels_downsampled = nda.rescale_nearest_neighbour(labels, (downsampling_size, downsampling_size))
    centers = np.array(nda.get_labels_centers(labels_downsampled))

    if centers.size == 0:
        return np.zeros_like(labels), np.zeros((downsampling_size, downsampling_size))

    array_centers_downsampled = np.zeros_like(labels_downsampled)
    array_centers_downsampled[centers[:, 0], centers[:, 1]] = 1

    centers = centers * (labels.shape[0] // downsampling_size)
    array_centers = np.zeros_like(labels)
    array_centers[centers[:, 0], centers[:, 1]] = 1

    return array_centers, array_centers_downsampled


def preprocess_tile(orthophoto: np.ndarray, dsm: np.ndarray, labels: np.ndarray, points: pd.DataFrame, downsampling_size=100, dsm_clip_height=30.0) -> np.ndarray:
    """
    :param orthophoto: np.ndarray with shape (H, W, 4) or (H, W, 3) representing the orthophoto image.
    :param dsm: np.ndarray with shape (H, W) representing the digital surface model (DSM or NDSM).
    :param labels: np.ndarray with shape (H, W) representing the instance segmentation labels.
    :param points: pd.DataFrame with columns `['axis-0', 'axis-1', 'code']` representing the tree species points.
    :param downsampling_size: int, size to which the labels will be downsampled.
    :param dsm_clip_height: float, height to which the DSM will be clipped.
    :return: dict containing the processed `{orthophoto, dsm, labels, labels_downsampled, species, centers, centers_downsampled}`.  
        - `orthophoto` is normalized to `[0.0, 1.0]` range  
        - `dsm` is clipped and rescaled to `[0.0, 1.0]` range where `1.0` equals `dsm_clip_height`
        - `labels` are relabeled to have unique values starting from `0` (does not use skimage's label function)  
        - `species` is a numpy array with the tree species codes `(int)` applied to the labels  
        - `centers` is a binary numpy array with the centers of the instances set to `1.0`
        - `centers_downsampled` is a numpy array with the centers of the downsampled instances
    """
    orthophoto = nda.normalize(orthophoto[..., :3])
    dsm = dsm[..., :1] if dsm.ndim > 2 else dsm
    dsm = nda.clip_rescale(dsm, dsm_clip_height)
    labels = nda.relabel(labels)
    species = apply_tree_species_codes(labels, points, get_tree_species_code)

    species_downsampled = nda.rescale_nearest_neighbour(species, (downsampling_size, downsampling_size))
    species_downsampled = median_filter(species_downsampled, size=3)

    labels_downsampled = nda.rescale_nearest_neighbour(labels, (downsampling_size, downsampling_size))
    labels_downsampled = median_filter(labels_downsampled, size=3)
    
    array_centers, array_centers_downsampled = get_centers(labels, downsampling_size)

    return {
        'orthophoto': orthophoto,
        'dsm': dsm,
        'labels': labels,
        'labels_downsampled': labels_downsampled,
        'species': species,
        'species_downsampled': species_downsampled,
        'centers': array_centers,
        'centers_downsampled': array_centers_downsampled
    }


def split_tile_dict(tile_dict: dict, downsampling_size=50) -> list:
    """
    Split a tile dictionary into 4 quadrants.
    Top-left, top-right, bottom-left, bottom-right.

    :param tile_dict: Dictionary containing tile data `{orthophoto, dsm, labels, species}`
    :param downsampling_size: Size to which the labels and species will be downsampled.
    :return: List of dictionaries, each containing the data for one quadrant. (`top-left`, `top-right`, `bottom-left`, `bottom-right`)
    """
    orthophoto = tile_dict['orthophoto']
    dsm = tile_dict['dsm']
    labels = tile_dict['labels']
    species = tile_dict['species']

    h, w = orthophoto.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    sub_tiles = []
    for i in range(2):
        for j in range(2):
            orthophoto_tile = orthophoto[i * mid_h:(i + 1) * mid_h, j * mid_w:(j + 1) * mid_w]
            dsm_tile = dsm[i * mid_h:(i + 1) * mid_h, j * mid_w:(j + 1) * mid_w]
            labels_tile = labels[i * mid_h:(i + 1) * mid_h, j * mid_w:(j + 1) * mid_w]
            species_tile = species[i * mid_h:(i + 1) * mid_h, j * mid_w:(j + 1) * mid_w]

            species_downsampled = nda.rescale_nearest_neighbour(species_tile, (downsampling_size, downsampling_size))
            species_downsampled = median_filter(species_downsampled, size=3)

            labels_downsampled = nda.rescale_nearest_neighbour(labels_tile, (downsampling_size, downsampling_size))
            labels_downsampled = median_filter(labels_downsampled, size=3)

            array_centers, array_centers_downsampled = get_centers(labels_tile, downsampling_size)

            sub_tiles.append({
                    'orthophoto': orthophoto_tile,
                    'dsm': dsm_tile,
                    'labels': labels_tile,
                    'labels_downsampled': labels_downsampled,
                    'species': species_tile,
                    'species_downsampled': species_downsampled,
                    'centers': array_centers,
                    'centers_downsampled': array_centers_downsampled
                })

    return sub_tiles


def save_tiles_dataset(tiles_dir):
    """
    Saves the tiles dataset by processing orthophotos, NDSMs, labels, and points.
    Splits each tile into four sub-tiles and saves them in a compressed numpy format.
    Each sub-tile is saved with a name indicating its position (nw, ne, sw, se).
    
    :param tiles_dir: Directory containing the `orthophoto`, `ndsm`, `labels`, and `points` directories.
    :return: None (saves files to folder `dataset in `tiles_dir`)
    """
    orthophoto_dir = os.path.join(tiles_dir, 'orthophoto')
    ndsm_dir = os.path.join(tiles_dir, 'ndsm')
    labels_dir = os.path.join(tiles_dir, 'labels')
    points_dir = os.path.join(tiles_dir, 'points')

    names = ['(nw)', '(ne)', '(sw)', '(se)']

    for file in os.listdir(orthophoto_dir):
        if not file.endswith('.tif'):
            continue

        orthophoto_path = os.path.join(orthophoto_dir, file)
        ndsm_path = os.path.join(ndsm_dir, file.replace('.tif', '.tif'))
        labels_path = os.path.join(labels_dir, file.replace('.tif', '.npz'))
        points_path = os.path.join(points_dir, file.replace('.tif', '.csv'))

        orthophoto = geo.to_ndarray(geo.open_geotiff(orthophoto_path))
        ndsm = geo.to_ndarray(geo.open_geotiff(ndsm_path))
        labels = nda.read_numpy(labels_path, npz_format='napari')
        points = pd.read_csv(points_path)

        tile_dict = preprocess_tile(
            orthophoto, 
            ndsm, 
            labels, 
            points, 
            downsampling_size=100, 
            dsm_clip_height=30.0
        )

        sub_tiles_dicts = split_tile_dict(tile_dict)

        for i, sub_tile in enumerate(sub_tiles_dicts):
            tile_name = file.replace('.tif', f' {names[i]}.npz')
            tile_path = os.path.join(tiles_dir, 'dataset', tile_name)
            np.savez_compressed(
                tile_path,
                orthophoto=sub_tile['orthophoto'],
                dsm=sub_tile['dsm'],
                labels=sub_tile['labels'],
                labels_downsampled=sub_tile['labels_downsampled'],
                species=sub_tile['species'],
                species_downsampled=sub_tile['species_downsampled'],
                centers=sub_tile['centers'],
                centers_downsampled=sub_tile['centers_downsampled']
            )
            print(f'Saved \'{tile_path}\'')
        print('')


# LOAD NPZ TILES FUNCTIONS

def _load_tiles_classes(data, n_classes=18, key='species', size: int = None) -> np.ndarray:
    """
    The class no1 is the unknown class, which is a mix of all species (present or absent in the classification)

    :param data: The data dictionary loaded from a .npz file
    :param n_classes: The number of classes in the dataset, including background and unknown
    :param key: The key in the data dictionary that contains the species labels
    :return: A numpy array of shape (height, width, n_classes) with one-hot encoded classes
    """
    classes = data[key]

    if size is not None and size > 0:
        classes = nda.rescale_nearest_neighbour(classes, (size, size))
        classes = median_filter(classes, size=3)

    one_hot = to_categorical(classes, n_classes)
    unknown = one_hot[..., 1] / (n_classes - 1)
    zeros = np.zeros_like(unknown)
    one_hot[..., 1] = zeros

    stacked = np.repeat(unknown[:, :, np.newaxis], (n_classes - 1), axis=2)
    stacked = np.concatenate((zeros[:, :, np.newaxis], stacked), axis=2)
    one_hot = one_hot + stacked

    return one_hot


def _load_tiles_rgbd(data, size: int = None) -> np.ndarray:
    """
    Loads the orthophoto and DSM from the data dictionary and stacks them along the last dimension.

    :param data: The data dictionary loaded from a .npz file
    :return: A numpy array of shape (height, width, 4) with RGB and DSM channels stacked
    """
    ortho = data['orthophoto']
    dsm = data['dsm']
    rgbd = np.dstack((ortho, dsm))

    if size is not None and size > 0:
        rgbd = nda.rescale_linear(rgbd, (size, size))

    return rgbd


def rgbd_rotation_augmentation(rgbd, classes):
    rotation_times = np.random.randint(0, 4)
    # print(f"Rotating {rotation_times} times")
    rgbd = np.rot90(rgbd, k=rotation_times, axes=(0, 1))
    classes = np.rot90(classes, k=rotation_times, axes=(0, 1))

    flip_horizontal = np.random.rand() > 0.5
    if flip_horizontal:
        # print("Flipping horizontally")
        rgbd = np.flip(rgbd, axis=1)
        classes = np.flip(classes, axis=1)

    flip_vertical = np.random.rand() > 0.5
    if flip_vertical:
        # print("Flipping vertically")
        rgbd = np.flip(rgbd, axis=0)
        classes = np.flip(classes, axis=0)

    return rgbd, classes


def rgbd_transformation_augmentation(rgbd, classes):
    ndsm_adjustment = np.random.rand() > 0.5
    if ndsm_adjustment:
        # print("Adjusting NDSM")
        ndsm = rgbd[:, :, 3]
        ndsm = ndsm * (1 + np.random.uniform(-0.1, 0.1))
        rgbd[:, :, 3] = np.clip(ndsm, 0.0, 1.0)

    contrast_adjustment = np.random.rand() > 0.25
    if contrast_adjustment:
        # print("Adjusting contrast")
        lower = np.random.uniform(0.2, 10.0)
        upper = np.random.uniform(90.0, 99.8)
        rgb = rgbd[:, :, :3]
        v_min, v_max = np.percentile(rgb, (lower, upper))
        rgb = exposure.rescale_intensity(rgb, in_range=(v_min, v_max))
        rgb = np.clip(rgb, 0.0, 1.0)
        rgbd[:, :, :3] = rgb

    gamma_adjustment = np.random.rand() > 0.25
    if gamma_adjustment:
        # print("Adjusting gamma")
        gamma = np.random.uniform(0.70, 0.98)
        gain = np.random.uniform(0.70, 0.98)
        rgbd[:, :, :3] = exposure.adjust_gamma(rgbd[:, :, :3], gamma, gain)

    add_noise = np.random.rand() > 0.25
    if add_noise:
        # print("Adding noise")
        var = np.random.uniform(0.001, 0.01)
        rgbd = random_noise(rgbd, mode='gaussian', var=var)
        rgbd = np.clip(rgbd, 0.0, 1.0)

    add_blur = np.random.rand() > 0.25
    if add_blur:
        # print("Adding blur")
        sigma = np.random.uniform(0.5, 1.5)
        for i in range(rgbd.shape[2]):
            l = ndimage.gaussian_filter(rgbd[:, :, i], sigma=sigma)
            l = np.clip(l, 0.0, 1.0)
            rgbd[:, :, i] = l

    return rgbd, classes


def tf_decode_item(item: tf.Tensor) -> int:
    if isinstance(item, tf.Tensor):
        return item.numpy().item()
    return item


def tf_decode_str(item: tf.Tensor) -> str:
    if isinstance(item, tf.Tensor):
        return item.numpy().decode("utf-8")
    return item


def load_tiles_data(path, n_classes, rgbd_size, classes_size, augmentation) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the RGB-D data and classes from a .npz file.

    :param path: The path to the .npz file containing the data
    :param n_classes: The number of classes in the dataset, including background and unknown
    :param rgbd_size: The size to which the RGB-D data will be resized (height and width)
    :param classes_size: The size to which the classes data will be resized (height and width)
    :param augmentation: The level of augmentation to apply (0 = none, 1 = rotations + mirror, 2 = rotation + mirror + transformation)
    :return: A tuple (rgbd, classes) where rgbd is a numpy array of shape (height, width, 4)
            and classes is a numpy array of shape (height, width, n_classes)
    """
    path = tf_decode_str(path)
    n_classes = tf_decode_item(n_classes)
    rgbd_size = tf_decode_item(rgbd_size)
    classes_size = tf_decode_item(classes_size)
    augmentation = tf_decode_item(augmentation)

    data = np.load(str(path))
    rgbd = _load_tiles_rgbd(data, size=rgbd_size)
    classes = _load_tiles_classes(data, n_classes, size=classes_size)

    if augmentation > 0: rgbd, classes = rgbd_rotation_augmentation(rgbd, classes)
    if augmentation > 1: rgbd, classes = rgbd_transformation_augmentation(rgbd, classes)

    return rgbd, classes


def init_tf_load_rgbd_tile(n_classes=18, rgbd_size=1024, classes_size=64, augmentation=2) -> Callable:
    """
    Returns a TensorFlow function for loading RGB-D data and classes from a .npz file.

    :param n_classes: The number of classes in the dataset, including background and unknown
    :param rgbd_size: The size to which the RGB-D data will be resized (height and width)
    :param classes_size: The size to which the classes data will be resized (height and width)
    :param augmentation: The level of augmentation to apply (0 = none, 1 = rotations + mirror, 2 = rotation + mirror + transformation)
    :return: A callable function that takes a file path as input and returns a tuple (rgbd, classes)
    """
    def tf_load_tile_npz(path) -> tuple[tf.Tensor, tf.Tensor]:
        """
        TensorFlow wrapper for loading RGB-D data and classes from a .npz file.
        """
        rgbd, classes = tf.py_function(
            func=load_tiles_data,
            inp=[path, n_classes, rgbd_size, classes_size, augmentation],
            Tout=(tf.float32, tf.float32)
        )
        rgbd.set_shape((rgbd_size, rgbd_size, 4))
        classes.set_shape((classes_size, classes_size, n_classes))
        return rgbd, classes
    
    return tf_load_tile_npz


def init_tf_load_rgb_tile(n_classes=18, rgb_size=1024, classes_size=64, augmentation=2) -> Callable:
    """
    Returns a TensorFlow function for loading RGB data and classes from a .npz file.

    :param n_classes: The number of classes in the dataset, including background and unknown
    :param rgb_size: The size to which the RGB data will be resized (height and width)
    :param classes_size: The size to which the classes data will be resized (height and width)
    :param augmentation: The level of augmentation to apply (0 = none, 1 = rotations + mirror, 2 = rotation + mirror + transformation)
    :return: A callable function that takes a file path as input and returns a tuple (rgb, classes)
    """
    def tf_load_tile_npz(path) -> tuple[tf.Tensor, tf.Tensor]:
        """
        TensorFlow wrapper for loading RGB data and classes from a .npz file.
        """
        rgbd, classes = tf.py_function(
            func=load_tiles_data,
            inp=[path, n_classes, rgb_size, classes_size, augmentation],
            Tout=(tf.float32, tf.float32)
        )
        rgb = rgbd[:, :, :3]
        rgb.set_shape((rgb_size, rgb_size, 3))
        classes.set_shape((classes_size, classes_size, n_classes))
        return rgb, classes
    
    return tf_load_tile_npz


def get_dataset_paths(dataset_dir, subset=None, *, shuffle=False) -> list[str]:
    """
    Retrieves paths to all .npz files in a specified subset directory.

    :param dataset_dir: The root directory containing the dataset
    :param subset: The subset directory (e.g., 'train', 'val', 'test')
    :param shuffle: Whether to shuffle the list of paths
    :return: A list of string paths to .npz files in the specified subset directory
    """
    dataset_paths = []

    if subset is not None:
        dataset_dir = os.path.join(dataset_dir, subset)

    for file_name in os.listdir(dataset_dir):
        if file_name.endswith('.npz'):
            dataset_paths.append(os.path.join(dataset_dir, file_name))

    if shuffle: np.random.shuffle(dataset_paths)
    return np.array(dataset_paths).tolist()


# CLASSES WEIGHTS

def _get_classes_weights(tile_path: str, key='species_downsampled', n_classes=18) -> np.ndarray:
    """
    Calculate the class weights from a single npz file.

    :param tile_path: Path to the .npz file containing class data.
    :param key: The key in the npz file that contains the class labels.
    :param n_classes: The number of classes to consider for weight calculation.
    :return: A numpy array of class weights of size (n_classes).
    """
    weights = np.zeros(n_classes)
    data = np.load(tile_path)
    classes = data[key]
    size = classes.size

    for i in range(n_classes):
        weights[i] = np.sum(classes == i) / size

    return weights


def get_dataset_weights(tiles_paths: list[str], key='species_downsampled', n_classes=18) -> np.ndarray:
    """
    Calculate the average class weights from npz files in the provided list of tile paths.

    :param tiles_paths: A list of paths to .npz files containing class data.
    :param key: The key in the npz file that contains the class labels.
    :param n_classes: The number of classes to consider for weight calculation.
    :return: A numpy array of average class weights of size (n_classes).
    """
    weights = np.zeros(n_classes)
    count = 0
    
    for tile_path in tiles_paths:
        weights += _get_classes_weights(tile_path, key, n_classes)
        count += 1

    weights /= count
    return weights


def rebalance_unknown_classes(weights: np.ndarray) -> np.ndarray:
    """
    Rebalance the weights to account for unknown classes at index 1.
    This function modifies the weights array such that the weight for the unknown class
    (index 1) is set to class_weight / (n_classes - 1), 
    the other classes values (except index 0) are increased by the same value

    :param weights: A numpy array of class weights.
    :return: A numpy array of rebalanced class weights.
    """
    n_classes = len(weights)
    unknown_weight = weights[1] / (n_classes - 1)
    weights[1] = 0
    unknown_weights = np.array([0] + [unknown_weight] * (n_classes - 1))
    weights += unknown_weights
    return weights


# MODELS

def jacard_coefficient(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + 1.0) / (union + 1.0)


def init_weighted_mean_iou(class_weights: list | np.ndarray, remove_bg_weight: bool) -> Callable:
    """
    Returns a weighted mean IoU metric function based on the provided class weights.

    :param class_weights: A list or numpy array of class weights.
    :param remove_bg_weight: If True, does not consider the background class (index 0) in the weighted mean IoU calculation.
    :return: A callable function that computes the weighted mean IoU.
    """
    def weighted_mean_iou(y_true, y_pred):
        num_classes = len(class_weights)

        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes)
        y_true = K.cast(y_true, "float32")
        y_pred = K.cast(y_pred, "float32")

        ious = []
        for i in range(num_classes):
            iou = jacard_coefficient(y_true[..., i], y_pred[..., i])
            ious.append(iou)

        if remove_bg_weight:
            ious = ious[1:]
            weights_no_bg = class_weights[1:]
        else:
            weights_no_bg = class_weights

        ious = tf.stack(ious)
        weights = K.constant(weights_no_bg, dtype="float32")
        weights = weights / K.sum(weights)
        return K.sum(ious * weights)

    return weighted_mean_iou


def DROPOUT_UNET_MODEL(*, input_shape=(1024, 1024, 4), n_classes=18, k_size=3, downsample_size=64, conv_count=16, dropout=0.0) -> Model:
    # adapted from https://youtu.be/jvZm8REF2KY
    conv_count = int(conv_count // 2) * 2
    inputs = Input(input_shape)

    #Contraction path
    c1 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(dropout)(c1) if dropout else c1
    c1 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    conv_count *= 2
    c2 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(dropout)(c2) if dropout else c2
    c2 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    conv_count *= 2
    c3 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(dropout)(c3) if dropout else c3
    c3 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    conv_count *= 2
    c4 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(dropout)(c4) if dropout else c4
    c4 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    conv_count *= 2
    c5 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(dropout)(c5) if dropout else c5
    c5 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    conv_count //= 2
    u6 = Conv2DTranspose(conv_count, (k_size, k_size), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(dropout)(c6) if dropout else c6
    c6 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    conv_count //= 2
    u7 = Conv2DTranspose(conv_count, (k_size, k_size), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(dropout)(c7) if dropout else c7
    c7 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    conv_count //= 2
    u8 = Conv2DTranspose(conv_count, (k_size, k_size), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(dropout)(c8) if dropout else c8
    c8 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    conv_count //= 2
    u9 = Conv2DTranspose(conv_count, (k_size, k_size), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(dropout)(c9) if dropout else c9
    c9 = Conv2D(conv_count, (k_size, k_size), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    downsampled = tf.image.resize(outputs, [downsample_size, downsample_size], method="nearest")
    model = Model(inputs=[inputs], outputs=[downsampled])
    
    return model


def UNET_MODEL(*, input_shape=(1024, 1024, 4), n_classes=18, k_size=3, downsample_size=64, conv_count=64) -> Model:
    conv_count = int(conv_count // 2) * 2
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(inputs)
    c1 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1) # 512x512

    conv_count *= 2
    c2 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(p1)
    c2 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2) # 256x256

    conv_count *= 2
    c3 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(p2)
    c3 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3) # 128x128

    conv_count *= 2
    c4 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(p3)
    c4 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4) # 64x64

    # Bottleneck
    conv_count *= 2
    c5 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(p4)
    c5 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(c5)

    # # Decoder
    conv_count //= 2
    u6 = UpSampling2D((2,2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(u6)
    c6 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(c6)

    conv_count //= 2
    u7 = UpSampling2D((2,2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(u7)
    c7 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(c7)

    conv_count //= 2
    u8 = UpSampling2D((2,2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(u8)
    c8 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(c8)

    conv_count //= 2
    u9 = UpSampling2D((2,2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(u9)
    c9 = Conv2D(conv_count, (k_size, k_size), activation='relu', padding='same')(c9)

    outputs = Conv2D(n_classes, (1,1), activation='softmax')(c9)
    downsampled = tf.image.resize(outputs, [downsample_size, downsample_size], method="nearest")
    model = Model(inputs=[inputs], outputs=[downsampled])

    return model


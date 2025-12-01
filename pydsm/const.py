########### SUB-DIRECTORIES ###########

ORTHOPHOTO_SUBDIR = 'orthophoto'
NDSM_SUBDIR = 'ndsm'
INSTANCE_LABELS_SUBDIR = 'labels'
SEMANTIC_POINTS_SUBDIR = 'points'
INSTANCES_TILES_SUBDIR = 'instances'
PREDICTION_INSTANCE_LABELS_SUBDIR = 'labels_pred'


########### CONSTANTS FOR WHOLE PROJECT ###########

CLIP_HEIGHT = 25.5  # in meters, for depth channel normalization in RGBD YOLO models
SEMANTIC_DICT = { 'BACKGROUND': 0, 'UNKNOWN': 1 }
DISTRIBUTION = [0.0, 1.0]


########### YOLO PREDICTIONS ###########

RGBD = True

CONFIDENCE_THRESHOLD = 0.2  # confidence threshold for YOLO predictions
IOU_THRESHOLD = 0.5  # IoU threshold for YOLO predictions

PIXEL_TOLERANCE = 24  # in pixels, tolerance for combining 2 instances
CIRCLE_TOLERANCE = 0.3  # in ratio, tolerance for combining 2 instances
MIN_HEIGHT = 1.0  # in meters, minimum height for predicted buildings

MIN_MASK_SIZE = 1024  # in pixels squared, minimum size for mask cleaning during preprocessing
REMOVE_CRACKS_SIZE = 5 # in pixels, size for morphological operation to remove cracks in masks during preprocessing


########### COPY PASTE ###########

MAX_INSTANCES = 10
MIN_GROUND_RATIO = 0.8
MAX_GROUND_HEIGHT = 2.0  # in meters


########### CUSTOM CONSTANTS BELLOW ###########

def tree_species_dict() -> dict:
    """
    20 classes total : 18 espèces + 1 arrière-plan + 1 inconnu
    """
    return {
        'BACKGROUND': 0,
        'UNKNOWN': 1,
        'ACSA': 2,
        'ACPL': 3,
        'ACSC': 4,
        'UL' : 5,
        'FR': 6,
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
        'PN': 18, # maybe out
        'JU': 19, # maybe out
    }


def equal_distribution(lenght: int = 2) -> list:
    return [0.0] + [1.0/(lenght-1)] * (lenght-1)


def tree_species_distribution() -> list:
    """
    Precalculated distribution of tree species to rebalance the dataset
    """
    return [0.0, 0.0, 0.034, 0.033, 0.061, 0.05, 0.051, 0.055, 0.057, 0.055, 0.057, 0.058, 0.059, 0.058, 0.06, 0.062, 0.062, 0.062, 0.063, 0.063]


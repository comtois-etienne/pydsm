########### SUB-DIRECTORIES ###########

ORTHOPHOTO_SUBDIR = 'orthophoto'
NDSM_SUBDIR = 'ndsm'
INSTANCE_LABELS_SUBDIR = 'labels'
SEMANTIC_POINTS_SUBDIR = 'points'
INSTANCES_TILES_SUBDIR = 'instances'
PREDICTED_TILES_SUBDIR = 'labels_pred'


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

MIN_MASK_SIZE = 2500  # in pixels squared, minimum size for mask cleaning during preprocessing
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


def tree_species_dict_2() -> dict:
    """
    20 classes total
    """
    def list_to_dict(l: list) -> dict:
        return {l[i]: i for i in range(len(l))}
    
    return list_to_dict([
        'BACKGROUND',
        'UNKNOWN',
        'ACPL',
        'ACSA',
        'GLTR',
        'FRPE',
        'UL',
        'PIPU',
        'SYRE',
        'CEOC',
        'QURU',
        'TICO',
        'MA',
        'PO',
        'GI',
        'GY',
        'PN',
        'AM',
    ])


def get_tile_mean_std() -> dict:
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


def equal_distribution(lenght: int = 2) -> list:
    return [0.0] + [1.0/(lenght-1)] * (lenght-1)


def tree_species_distribution() -> list:
    """
    Precalculated distribution of tree species to rebalance the dataset
    """
    return [0.0, 0.0, 0.039, 0.043, 0.046, 0.054, 0.059, 0.065, 0.065, 0.066, 0.068, 0.069, 0.068, 0.071, 0.071, 0.072, 0.072, 0.072]


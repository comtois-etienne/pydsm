########### CONSTANTS FOR WHOLE PROJECT ###########

CLIP_HEIGHT = 30.0  # in meters, for depth channel normalization in RGBD YOLO models


########### YOLO PREDICTIONS ###########

CONFIDENCE_THRESHOLD = 0.2  # confidence threshold for YOLO predictions
IOU_THRESHOLD = 0.5  # IoU threshold for YOLO predictions

PIXEL_TOLERANCE = 20  # in pixels, tolerance for combining 2 instances
CIRCLE_TOLERANCE = 0.3  # in ratio, tolerance for combining 2 instances
MIN_HEIGHT = 1.0  # in meters, minimum height for predicted buildings

MIN_MASK_SIZE = 400  # in pixels squared, minimum size for mask cleaning during preprocessing
REMOVE_CRACKS_SIZE = 5 # in pixels, size for morphological operation to remove cracks in masks during preprocessing

# todo add subfolders

def display_constants():
    print("CLIP_HEIGHT:", CLIP_HEIGHT)
    print("CONFIDENCE_THRESHOLD:", CONFIDENCE_THRESHOLD)
    print("IOU_THRESHOLD:", IOU_THRESHOLD)
    print("PIXEL_TOLERANCE:", PIXEL_TOLERANCE)
    print("CIRCLE_TOLERANCE:", CIRCLE_TOLERANCE)
    print("MIN_HEIGHT:", MIN_HEIGHT)
    print("MIN_MASK_SIZE:", MIN_MASK_SIZE)
    print("REMOVE_CRACKS_SIZE:", REMOVE_CRACKS_SIZE)
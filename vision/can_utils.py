import math
import numpy as np
from colors import CAN_CLASS_NAMES
from config import CAN_DIAMETER, CAN_CONFIDENCE_THRESHOLD, CAN_MIN_AREA_PIXELS
from .pixelTo3D import transform_uv_to_xy
from .mask_utils import getSmoothRegionFromMask, yoloMaskToBinary


def getBottomCenterPixel(mask):
    """
    Gets the bottom center pixel from a segmentation mask.
    Uses mean u coordinate across entire mask and bottom-most v coordinate.

    Args:
        mask: Binary mask (H, W) with values 0 or 255

    Returns:
        tuple: (x, y) coordinates of bottom center pixel, or None if mask is empty
            x: mean u coordinate across entire mask
            y: bottom-most v coordinate
    """
    # Get smooth region from mask
    roi_mask = getSmoothRegionFromMask(mask)

    # Find all non-zero pixels
    non_zero = np.argwhere(roi_mask > 0)

    if len(non_zero) == 0:
        return None

    # non_zero is array of [y, x] coordinates
    # Find maximum y value (bottom-most pixel)
    max_y = np.max(non_zero[:, 0])

    # Calculate mean x coordinate across entire mask
    x_coords = non_zero[:, 1]
    center_x = int(np.mean(x_coords))

    # Return mean x across entire mask and bottom y
    return (center_x, max_y)


def getCans(result, image, is_top=True):
    """
    Extracts can locations from YOLO results and transforms to xy coordinates.

    Args:
        result: YOLO result object from inference
        image: Original BGR image (used for mask processing)

    Returns:
        tuple: (can_locations_xy, class_names)
            - can_locations_xy: List of [x, y] in mm (ground plane coordinates)
            - class_names: List of strings with class names for each can
    """
    can_locations_xy = []
    class_names = []

    if result.masks is None:
        return can_locations_xy, class_names

    for i, mask_orig in enumerate(result.masks):
        # Get class name and confidence for this detection
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]
        confidence = float(result.boxes.conf[i])

        # Only process can detections
        if class_name not in CAN_CLASS_NAMES:
            continue

        # Skip cans with confidence below threshold
        if confidence < CAN_CONFIDENCE_THRESHOLD:
            continue

        # Convert mask to grayscale image
        binary_mask = yoloMaskToBinary(mask_orig, image)

        # Calculate mask area (number of non-zero pixels)
        mask_area = np.count_nonzero(binary_mask)

        # Skip cans with area below threshold (likely specs/noise)
        if mask_area < CAN_MIN_AREA_PIXELS:
            continue

        # Get bottom center pixel coordinates
        bottom_center = getBottomCenterPixel(binary_mask)

        if bottom_center is None:
            continue

        u, v = bottom_center  # pixel coordinates

        # Transform to ground plane coordinates
        xy = transform_uv_to_xy(u, v, is_top)

        if xy is None:
            continue

        # this is the front center location
        x, y = xy
        # get the center center location
        theta = math.atan2(y, x)
        x += CAN_DIAMETER / 2 * math.cos(theta)
        y += CAN_DIAMETER / 2 * math.sin(theta)
        # add the info
        can_locations_xy.append([x, y])
        class_names.append(class_name)

    return can_locations_xy, class_names

from mask_utils import getSmoothRegionFromMask, yoloMaskToBinary
import numpy as np
from pixelTo3D import transform_uv_to_xy


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


def getCans(result, image):
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
        # Get class name for this detection
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]

        # Only process can detections
        if class_name not in ['Green Can', 'Golden Can', 'Red Can']:
            continue

        # Convert mask to grayscale image
        binary_mask = yoloMaskToBinary(mask_orig, image)

        # Get bottom center pixel coordinates
        bottom_center = getBottomCenterPixel(binary_mask)

        if bottom_center is None:
            continue

        u, v = bottom_center  # pixel coordinates

        # Transform to ground plane coordinates
        xy = transform_uv_to_xy(u, v)
        print(f"Can at pixel ({u}, {v}) -> xy: {xy}")

        if xy is None:
            continue

        x, y = xy
        can_locations_xy.append([x, y])
        class_names.append(class_name)

    return can_locations_xy, class_names

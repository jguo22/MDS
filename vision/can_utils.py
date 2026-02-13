import math
import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple
from colors import CAN_CLASS_NAMES
from config import CAN_DIAMETER, CAN_MIN_AREA_PIXELS
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


def getCans(result, image,
            is_top) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Extracts can locations from YOLO results and transforms to xy coordinates.

    Args:
        result: YOLO result object from inference
        image: Original BGR image (used for mask processing)

    Returns:
        tuple: (can_locations_xy, class_names)
            - can_locations_xy: List of (x, y) in mm (ground plane coordinates)
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
        can_locations_xy.append((x, y))
        class_names.append(class_name)

    return can_locations_xy, class_names


def is_hull_overlap_with_target_rect(
    convex_hull: np.ndarray,
    rectangle: np.ndarray,
    min_intersection_over_rect: float,
    max_segmentation_over_rect: float,
) -> bool:
    """
    Check whether a convex hull sufficiently overlaps a fixed image-space rectangle.

    The rectangle is defined in (u, v) pixel coordinates by ``TARGET_RECT_UV``:
    (295, 480), (295, 260), (540, 260), (540, 480).

    This function computes:

    - the area of the convex hull in pixel space (``hull_area``)
    - the area of the fixed rectangle (``rect_area``)
    - the intersection area between the hull and the rectangle
      (``intersection_area``)
    - the ratio ``intersection_area / rect_area``
    - the ratio ``hull_area / rect_area``

    It returns ``True`` only if both of the following hold:

    - ``intersection_area / rect_area >= min_intersection_over_rect``
    - ``hull_area / rect_area <= max_segmentation_over_rect``

    Args:
        convex_hull: Convex hull vertices in pixel (u, v) coordinates, as:
            - numpy array with shape (N, 1, 2) or (N, 2)
        min_intersection_over_rect: Minimum required ratio of
            ``intersection_area / rect_area`` (0.0–1.0).
        max_segmentation_over_rect: Maximum allowed ratio of
            ``hull_area / rect_area`` (0.0–1.0).

    Returns:
        bool: ``True`` if the hull overlaps the rectangle enough and the hull
        does not cover too much of the rectangle, ``False`` otherwise.
    """
    # Basic validation
    if convex_hull is None:
        return False

    try:
        # Handle both (N, 1, 2) and (N, 2) shapes
        if isinstance(convex_hull, np.ndarray):
            if convex_hull.ndim == 3:
                convex_hull = convex_hull.reshape(-1, 2)
        else:
            # Unsupported type
            return False

        if convex_hull.shape[0] < 3:
            # Need at least a triangle to have non-zero area
            return False

        hull_poly = Polygon(convex_hull)
        rect_poly = Polygon(rectangle)

        # Validate polygons
        if not hull_poly.is_valid or not rect_poly.is_valid:
            return False

        hull_area = hull_poly.area
        rect_area = rect_poly.area

        if hull_area <= 0.0 or rect_area <= 0.0:
            return False

        intersection = hull_poly.intersection(rect_poly)
        intersection_area = intersection.area

        if intersection_area <= 0.0:
            return False

        intersection_over_rect = intersection_area / rect_area
        segmentation_over_rect = hull_area / rect_area

        return (
            intersection_over_rect >= min_intersection_over_rect
            and segmentation_over_rect <= max_segmentation_over_rect
        )

    except Exception:
        # On any geometric/numeric error, treat as non-overlapping
        return False

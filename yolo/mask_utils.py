"""
Mask processing utilities for segmentation refinement.
"""
import cv2 as cv
import numpy as np
from pathlib import Path
from ultralytics.engine.results import Masks


SCRIPT_DIR = Path(__file__).parent.absolute()

# Global parameter dictionary for interactive tuning
params = {
    'dom_sat_min': 70,
    'dom_val_min': 140,
    'abs_sat_min': 80,
    'abs_val_min': 120,
    'not_black_thresh': 100,
    'rel_sat_diff': -5,
    'rel_val_diff': -10,
    'hue_tolerance': 15,
    'blur_size': 21,
    'erode_size': 15,
}


def enforce_odd(value):
    """Ensures kernel sizes are odd numbers."""
    return value if value % 2 == 1 else value - 1


def maskToFilledContours(mask):
    """
    Finds outer contours and fills them completely.

    Args:
        mask: Binary mask (H, W) with values 0 or 255

    Returns:
        Filled mask with all contours filled
    """
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv.drawContours(filled, contours, -1, 255, -1)  # thickness=-1 fills

    return filled


def maskToFilledBiggestContour(mask):
    """
    Finds the largest contour and fills it.

    Args:
        mask: Binary mask (H, W) with values 0 or 255

    Returns:
        Filled mask with only the largest contour
    """
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(mask)
    if len(contours) == 0:
        print("no contour")
    else:
        contour = max(contours, key=cv.contourArea)
        cv.drawContours(filled, [contour], 0, 255, -1)

    return filled


def maskToConvexRegion(mask):
    """
    Converts mask to filled convex hull region encompassing ALL contours.

    Args:
        mask: Binary mask (H, W) with values 0 or 255

    Returns:
        Filled convex hull mask (H, W)
    """
    # Find all contours
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros_like(mask)

    # Combine ALL contour points into single array
    all_points = np.vstack(contours)

    # Compute single convex hull around all points
    hull = cv.convexHull(all_points)

    # Create filled mask from hull
    filled = np.zeros_like(mask)
    cv.drawContours(filled, [hull], 0, 255, -1)

    return filled


def getSmoothRegionFromMask(mask):
    """
    Erodes mask and returns convex hull region.

    Args:
        mask: Binary mask (H, W) with values 0 or 255

    Returns:
        Smoothed convex region mask
    """
    size = 5
    kernel = np.ones((size, size), np.uint8)
    eroded = cv.erode(mask, kernel)

    roi_mask = maskToConvexRegion(eroded)

    return roi_mask


def calculateQuadFromMask(segmentation_mask: np.ndarray):
    """
    Improved Douglas-Peucker with area-preserving score.

    Args:
        segmentation_mask: Binary mask (H, W) with values 0 or 255

    Returns:
        Quadrilateral vertices as numpy array (4, 1, 2), or None if no contour found
    """
    _, binary_mask = cv.threshold(
        segmentation_mask, 127, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(
        binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv.contourArea)
    hull = cv.convexHull(largest_contour)
    original_area = cv.contourArea(hull)
    perimeter = cv.arcLength(hull, True)

    # Try many epsilon values
    best_quad = None
    best_score = float('inf')

    for eps_factor in np.linspace(0.005, 0.15, 50):
        epsilon = eps_factor * perimeter
        approx = cv.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4:
            # Score based on area difference (want to preserve area)
            approx_area = cv.contourArea(approx)
            area_diff = abs(original_area - approx_area)

            # Also penalize very small angles
            angles = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % 4][0]
                p3 = approx[(i + 2) % 4][0]

                v1 = p1 - p2
                v2 = p3 - p2

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)
                                              * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)

            min_angle = np.min(angles)
            angle_penalty = 0 if min_angle > np.radians(30) else 1e6

            score = area_diff + angle_penalty

            if score < best_score:
                best_score = score
                best_quad = approx

    if best_quad is not None:
        return best_quad

    # Fallback
    rect = cv.minAreaRect(hull)
    return cv.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)


def fixSegmentation(image, mask):
    """
    Removes black border and adds missing colored tape.
    Uses relative saturation - robust to lighting changes.
    Uses global params dictionary for all thresholds.

    Args:
        image: BGR image
        mask: Binary mask (H, W) with values 0 or 255

    Returns:
        Fixed mask with convex hull of detected colored regions
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # Erode to remove black border
    erode_size = enforce_odd(params['erode_size'])
    kernel_erode = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (erode_size, erode_size))
    mask_eroded = cv.erode(mask, kernel_erode)

    # Find dominant hue from bright saturated pixels
    h_masked = cv.bitwise_and(h, h, mask=mask_eroded)
    s_masked = cv.bitwise_and(s, s, mask=mask_eroded)
    v_masked = cv.bitwise_and(v, v, mask=mask_eroded)

    valid = (
        s_masked > params['dom_sat_min']) & (
        v_masked > params['dom_val_min'])
    if np.any(valid):
        dominant_hue = np.median(h_masked[valid])
    else:
        return mask

    # Find similar colored pixels using relative saturation
    roi = maskToConvexRegion(mask)
    s_roi = cv.bitwise_and(s, s, mask=roi)
    h_roi = cv.bitwise_and(h, h, mask=roi)
    v_roi = cv.bitwise_and(v, v, mask=roi)

    # Compute local average saturation and value
    blur_size = enforce_odd(params['blur_size'])
    s_local = cv.blur(s_roi, (blur_size, blur_size))
    v_local = cv.blur(v_roi, (blur_size, blur_size))

    # Relative comparisons (robust to lighting)
    s_diff = s_roi.astype(np.int16) - s_local.astype(np.int16)
    saturated = s_diff > params['rel_sat_diff']

    # Brighter than local neighborhood
    v_diff = v_roi.astype(np.int16) - v_local.astype(np.int16)
    bright = v_diff > params['rel_val_diff']

    # Require minimum absolute saturation and brightness
    min_saturation = s_roi > params['abs_sat_min']
    min_brightness = v_roi > params['abs_val_min']

    # Explicitly exclude very dark pixels
    not_black = v_roi > params['not_black_thresh']

    # Hue matching
    hue_diff = np.abs(h_roi.astype(np.int16) - dominant_hue)
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    similar_hue = hue_diff < params['hue_tolerance']

    # Combine all filters
    tape_mask = (saturated & bright & min_saturation & min_brightness &
                 not_black & similar_hue).astype(np.uint8) * 255

    # Clean up
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    tape_mask = cv.morphologyEx(tape_mask, cv.MORPH_CLOSE, kernel)

    # Erode to remove remaining noise
    kernel_erode_final = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3))
    tape_mask = cv.erode(tape_mask, kernel_erode_final)

    # Return convex hull
    return maskToConvexRegion(tape_mask)


def yoloMaskToBinary(mask_orig: Masks, image: np.ndarray):
    # Convert mask to grayscale image
    mask_array = mask_orig.data[0].cpu().numpy()
    mask_uint8 = (mask_array * 255).astype(np.uint8)

    # Resize mask to match original image size
    mask_resized = cv.resize(
        mask_uint8, (image.shape[1], image.shape[0]))

    _, binary_mask = cv.threshold(
        mask_resized, 127, 255, cv.THRESH_BINARY)

    return binary_mask

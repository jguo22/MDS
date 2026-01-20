import cv2 as cv
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import json


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


def load_parameters():
    """Load parameters from JSON file if it exists."""
    params_file = SCRIPT_DIR / "segment_params.json"
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                loaded = json.load(f)
                params.update(loaded)
                print(f"Loaded saved parameters from {params_file}")
                return True
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return False
    return False


def save_parameters():
    """Save current parameters to JSON file."""
    params_file = SCRIPT_DIR / "segment_params.json"
    try:
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Parameters saved to {params_file}")
        return True
    except Exception as e:
        print(f"Error saving parameters: {e}")
        return False


def wait_for_quit():
    """Waits for keypress. Returns True if 'q' pressed, False otherwise."""
    if cv.waitKey(0) & 0xFF == ord('q'):
        raise


def compute_iou(mask1, mask2):
    """
    Computes Intersection over Union between two binary masks.

    Args:
        mask1, mask2: Binary masks (H, W) with values 0 or 255

    Returns:
        IoU score (0.0 to 1.0)
    """
    intersection = cv.bitwise_and(mask1, mask2)
    union = cv.bitwise_or(mask1, mask2)

    intersection_area = np.count_nonzero(intersection)
    union_area = np.count_nonzero(union)

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def click_hsv(event, x, y, flags, param):
    """Mouse callback to print HSV values at clicked point"""
    if event == cv.EVENT_LBUTTONDOWN:
        image, hsv = param
        h, s, v = hsv[y, x]
        bgr = image[y, x]
        print(f"Clicked ({x}, {y}): H={h}, S={s}, V={v} | BGR={bgr}")


def maskToFilledContours(mask):
    # Find outer contour and fill it completely
    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv.drawContours(filled, contours, -1, 255, -1)  # thickness=-1 fills

    return filled


def maskToFilledBiggestContour(mask):
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
    size = 5
    kernel = np.ones((size, size), np.uint8)
    eroded = cv.erode(mask, kernel)

    roi_mask = maskToConvexRegion(eroded)

    return roi_mask


def getBottomCenterPixel(mask):
    """
    Gets the bottom center pixel from a segmentation mask.

    Args:
        mask: Binary mask (H, W) with values 0 or 255

    Returns:
        tuple: (x, y) coordinates of bottom center pixel, or None if mask is empty
    """
    # Get smooth region from mask
    roi_mask = getSmoothRegionFromMask(mask)

    # Find all non-zero pixels
    non_zero = np.argwhere(roi_mask > 0)

    if len(non_zero) == 0:
        return None

    # non_zero is array of [y, x] coordinates
    # Find maximum y value (bottom-most row)
    max_y = np.max(non_zero[:, 0])

    # Get all pixels in the bottom row
    bottom_pixels = non_zero[non_zero[:, 0] == max_y]

    # Find center x coordinate among bottom pixels
    x_coords = bottom_pixels[:, 1]
    center_x = int(np.mean(x_coords))

    return (center_x, max_y)


def getQuadCenter(quad):
    """
    Calculates the center point of a quadrilateral.

    Args:
        quad: Quadrilateral vertices as numpy array with shape (4, 1, 2)
              Format: [[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]]

    Returns:
        tuple: (center_x, center_y) as integers
    """
    # Reshape from (4, 1, 2) to (4, 2) for easier processing
    points = quad.reshape(4, 2)

    # Calculate mean of all x and y coordinates
    center_x = int(np.mean(points[:, 0]))
    center_y = int(np.mean(points[:, 1]))

    return (center_x, center_y)


def getClosestZonesByColor(quadrilaterals, class_names):
    """
    Finds the closest zone of each color from detected quadrilaterals.

    Proximity is determined by y-coordinate: zones with higher y-values
    (closer to bottom of image) are physically closer to the robot.

    Args:
        quadrilaterals: List of numpy arrays (N, 1, 2) representing quad corners
        class_names: List of strings with class names (e.g., 'Green Zone', 'Red Zone', 'Golden Zone')

    Returns:
        tuple: (closest_quadrilaterals, closest_class_names)
            - closest_quadrilaterals: List of numpy arrays (N, 1, 2) for closest zones
            - closest_class_names: List of strings with class names for each closest zone
            Returns one quad per color detected, in arbitrary order.

    Raises:
        ValueError: If quadrilaterals and class_names have different lengths
    """
    if len(quadrilaterals) != len(class_names):
        raise ValueError(
            "quadrilaterals and class_names must have the same length")

    if len(quadrilaterals) == 0:
        return [], []

    # Group zones by color
    zones_by_color = {}
    for quad, class_name in zip(quadrilaterals, class_names):
        # Extract color from class name (e.g., 'Green Zone' -> 'Green')
        color = class_name.split()[0]

        if color not in zones_by_color:
            zones_by_color[color] = []

        zones_by_color[color].append((quad, class_name))

    # Find closest zone of each color
    # Closer zones have higher y-coordinate (bottom of image = closer to robot)
    closest_quadrilaterals = []
    closest_class_names = []

    for color, zones in zones_by_color.items():
        closest_quad = None
        closest_name = None
        max_y = -1

        for quad, class_name in zones:
            center_x, center_y = getQuadCenter(quad)

            # Zone with highest y-coordinate is closest to robot
            if center_y > max_y:
                max_y = center_y
                closest_quad = quad
                closest_name = class_name

        closest_quadrilaterals.append(closest_quad)
        closest_class_names.append(closest_name)

    return closest_quadrilaterals, closest_class_names


def annotate_poly(image, polygon, color=(0, 0, 255)):
    """
    Draws polygon on image
    Annotates in place

    Args:
        image: Image to annotate (numpy array)
              Shape: (H, W, 3) for BGR color image
        polygon: Polygon contour (N, 1, 2) numpy array
        color: Color for quadrilateral (BGR tuple), default red (0, 0, 255)

    Returns:
        None
    """

    # Draw the polygon
    cv.drawContours(image, [polygon], 0, color, 1)

    # Draw corner points
    for point in polygon:
        cv.circle(image, tuple(point[0]), 5, color, -1)

    return image


def calculateQuadFromMask(segmentation_mask: np.ndarray):
    """
    Improved Douglas-Peucker with area-preserving score.
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


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Creates a colored overlay of the mask on the image.

    Args:
        image: BGR image
        mask: Binary mask (H, W) with values 0 or 255
        color: BGR color tuple (default: green)
        alpha: Transparency of overlay (0.0 to 1.0, default: 0.5)

    Returns:
        Blended image with mask overlay
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    result = cv.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result


def fixSegmentation(image, mask):
    """
    Removes black border and adds missing colored tape.
    Uses relative saturation - robust to lighting changes.
    Uses global params dictionary for all thresholds.
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # Erode to remove black border
    erode_size = enforce_odd(params['erode_size'])
    kernel_erode = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (erode_size, erode_size))
    mask_eroded = cv.erode(mask, kernel_erode)

    # Find dominant hue from bright saturated pixels (use absolute threshold
    # here)
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
    # More saturated than local neighborhood = colored tape
    s_diff = s_roi.astype(np.int16) - s_local.astype(np.int16)
    saturated = s_diff > params['rel_sat_diff']

    # Brighter than local neighborhood (excludes dark interior/shadows)
    v_diff = v_roi.astype(np.int16) - v_local.astype(np.int16)
    bright = v_diff > params['rel_val_diff']

    # Also require minimum absolute saturation and brightness (prevent
    # gray/black regions)
    min_saturation = s_roi > params['abs_sat_min']
    min_brightness = v_roi > params['abs_val_min']

    # Explicitly exclude very dark pixels (black has v < 50 typically)
    not_black = v_roi > params['not_black_thresh']

    # Hue matching
    hue_diff = np.abs(h_roi.astype(np.int16) - dominant_hue)
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    similar_hue = hue_diff < params['hue_tolerance']

    # Combine all filters - must pass ALL conditions
    tape_mask = (saturated & bright & min_saturation & min_brightness &
                 not_black & similar_hue).astype(np.uint8) * 255

    # Clean up
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    tape_mask = cv.morphologyEx(tape_mask, cv.MORPH_CLOSE, kernel)

    # Erode to remove remaining noise
    kernel_erode_final = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (3, 3))  # radius 1
    tape_mask = cv.erode(tape_mask, kernel_erode_final)

    # Return convex hull
    return maskToConvexRegion(tape_mask)


def setup_control_panel():
    """Creates control panel window with trackbars for all parameters."""
    window_name = "Control Panel"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 400, 800)

    # Trackbars with proper ranges
    # Note: For negative values, we map trackbar 0-100 to actual -50 to +50
    cv.createTrackbar(
        "dom_sat_min",
        window_name,
        params['dom_sat_min'],
        255,
        lambda x: None)
    cv.createTrackbar(
        "dom_val_min",
        window_name,
        params['dom_val_min'],
        255,
        lambda x: None)
    cv.createTrackbar(
        "abs_sat_min",
        window_name,
        params['abs_sat_min'],
        255,
        lambda x: None)
    cv.createTrackbar(
        "abs_val_min",
        window_name,
        params['abs_val_min'],
        255,
        lambda x: None)
    cv.createTrackbar(
        "not_black_thresh",
        window_name,
        params['not_black_thresh'],
        255,
        lambda x: None)

    # Relative thresholds: map -50 to +50 → trackbar 0 to 100
    cv.createTrackbar(
        "rel_sat_diff",
        window_name,
        params['rel_sat_diff'] + 50,
        100,
        lambda x: None)
    cv.createTrackbar(
        "rel_val_diff",
        window_name,
        params['rel_val_diff'] + 50,
        100,
        lambda x: None)

    cv.createTrackbar(
        "hue_tolerance",
        window_name,
        params['hue_tolerance'],
        90,
        lambda x: None)
    cv.createTrackbar(
        "blur_size",
        window_name,
        params['blur_size'],
        51,
        lambda x: None)
    cv.createTrackbar(
        "erode_size",
        window_name,
        params['erode_size'],
        31,
        lambda x: None)

    return window_name


def read_trackbar_values(window_name):
    """Reads trackbar values and updates global params."""
    params['dom_sat_min'] = cv.getTrackbarPos("dom_sat_min", window_name)
    params['dom_val_min'] = cv.getTrackbarPos("dom_val_min", window_name)
    params['abs_sat_min'] = cv.getTrackbarPos("abs_sat_min", window_name)
    params['abs_val_min'] = cv.getTrackbarPos("abs_val_min", window_name)
    params['not_black_thresh'] = cv.getTrackbarPos(
        "not_black_thresh", window_name)

    # Map relative thresholds: trackbar 0-100 → actual -50 to +50
    params['rel_sat_diff'] = cv.getTrackbarPos(
        "rel_sat_diff", window_name) - 50
    params['rel_val_diff'] = cv.getTrackbarPos(
        "rel_val_diff", window_name) - 50

    params['hue_tolerance'] = cv.getTrackbarPos("hue_tolerance", window_name)

    # Enforce odd values for kernel sizes
    blur_raw = cv.getTrackbarPos("blur_size", window_name)
    params['blur_size'] = max(5, enforce_odd(blur_raw))  # Minimum 5

    erode_raw = cv.getTrackbarPos("erode_size", window_name)
    params['erode_size'] = max(3, enforce_odd(erode_raw))  # Minimum 3


def reset_trackbars(window_name):
    """Resets all trackbars to default values."""
    defaults = {
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

    cv.setTrackbarPos("dom_sat_min", window_name, defaults['dom_sat_min'])
    cv.setTrackbarPos("dom_val_min", window_name, defaults['dom_val_min'])
    cv.setTrackbarPos("abs_sat_min", window_name, defaults['abs_sat_min'])
    cv.setTrackbarPos("abs_val_min", window_name, defaults['abs_val_min'])
    cv.setTrackbarPos(
        "not_black_thresh",
        window_name,
        defaults['not_black_thresh'])
    cv.setTrackbarPos(
        "rel_sat_diff",
        window_name,
        defaults['rel_sat_diff'] + 50)
    cv.setTrackbarPos(
        "rel_val_diff",
        window_name,
        defaults['rel_val_diff'] + 50)
    cv.setTrackbarPos("hue_tolerance", window_name, defaults['hue_tolerance'])
    cv.setTrackbarPos("blur_size", window_name, defaults['blur_size'])
    cv.setTrackbarPos("erode_size", window_name, defaults['erode_size'])

    params.update(defaults)
    print("Parameters reset to defaults")


MODEL = YOLO(str(SCRIPT_DIR / 'last.pt'))


def segmentImage(image):
    # model returns array of results
    # here, we only have one image so its an array of size 1
    result = MODEL(image)[0]
    return result


def getQuadrilateralsAndClasses(result, image):
    """
    Extracts quadrilaterals and class names from YOLO segmentation results.

    Args:
        result: YOLO result object from inference
        image: Original BGR image (used for fixSegmentation)

    Returns:
        tuple: (quadrilaterals, class_names)
            - quadrilaterals: List of numpy arrays (N, 1, 2) representing quad corners
            - class_names: List of strings with class names for each quad
    """
    quadrilaterals = []
    class_names = []

    if result.masks is None:
        return quadrilaterals, class_names

    for i, mask_orig in enumerate(result.masks):
        # Convert mask to grayscale image
        mask_array = mask_orig.data[0].cpu().numpy()
        mask_uint8 = (mask_array * 255).astype(np.uint8)

        # Resize mask to match original image size
        mask_resized = cv.resize(
            mask_uint8, (image.shape[1], image.shape[0]))

        # Apply fixSegmentation to improve mask quality
        tape_mask = fixSegmentation(image, mask_resized)

        # Calculate quadrilateral from fixed mask
        quad = calculateQuadFromMask(tape_mask)

        if quad is not None:
            # Get class name for this detection
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]

            if (class_name in ['Green Zone', 'Golden Zone', 'Red Zone']):
                quadrilaterals.append(quad)
                class_names.append(class_name)

    return quadrilaterals, class_names


def main():
    image_path = str(SCRIPT_DIR / "test.jpg")
    print(f"Loading image: {image_path}")

    image = cv.imread(image_path)
    if image is None:
        raise Exception("no image")

    print("Running segmentation...")
    result = segmentImage(image)

    # Example: Use getQuadrilateralsAndClasses to extract results
    quads, classes = getQuadrilateralsAndClasses(result, image)
    print(f"\nFound {len(quads)} objects:")
    for i, (quad, class_name) in enumerate(zip(quads, classes)):
        print(f"  {i}: {class_name} - {len(quad)} corners")

    closest_quads, closest_classes = getClosestZonesByColor(quads, classes)
    print(f"\nClosest zones by color:")
    for quad, class_name in zip(closest_quads, closest_classes):
        center = getQuadCenter(quad)
        print(f"  {class_name} at center {center}")

    centers = [getQuadCenter(center) for center in closest_quads]
    print(centers)

    return

    # Display original segmentation
    annotated_frame = result.plot(boxes=False)
    cv.imshow("Original Segmentation", annotated_frame)

    # Load saved parameters if they exist
    load_parameters()

    # Setup control panel with trackbars
    control_window = setup_control_panel()

    # Print help message
    print("\nControl Panel created with current parameters")
    print("Controls:")
    print("  Q = Quit")
    print("  R = Reset Parameters to defaults")
    print("  S = Save Parameters to file")
    print("Adjust trackbars to tune segmentation in real-time\n")

    # Colors for different objects
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]

    # Event loop for interactive parameter tuning
    while True:
        # Read current trackbar values and update params
        read_trackbar_values(control_window)

        # Create fresh images for this iteration
        all_quads_image = image.copy()
        all_quads_image2 = image.copy()
        all_quads_image3 = image.copy()

        # Process all masks with current parameters
        if result.masks is not None:
            for i, mask_orig in enumerate(result.masks):
                # Convert mask to grayscale image
                mask_array = mask_orig.data[0].cpu().numpy()
                mask_uint8 = (mask_array * 255).astype(np.uint8)

                # Resize mask to match original image size
                mask_resized = cv.resize(
                    mask_uint8, (image.shape[1], image.shape[0]))

                # Apply fixSegmentation with current params
                tape_mask = fixSegmentation(image, mask_resized)

                quad = calculateQuadFromMask(mask_resized)
                quad2 = calculateQuadFromMask(tape_mask)

                if quad is None or quad2 is None:
                    continue

                # Annotate with unique color
                color = colors[i % len(colors)]
                annotate_poly(all_quads_image, quad, color)
                annotate_poly(all_quads_image2, quad2, color)

                # Overlay segmentation
                all_quads_image3 = overlay_mask(
                    all_quads_image3, tape_mask, color=color, alpha=0.4)

            # Display all windows
            cv.imshow("All Quadrilaterals", all_quads_image)
            cv.imshow("All Quadrilaterals 2", all_quads_image2)
            cv.imshow("All Segmentations", all_quads_image3)
        else:
            print("No segmentation masks found in the image")
            break

        # Handle keyboard input
        key = cv.waitKey(100) & 0xFF  # 100ms refresh = ~10 FPS
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('r'):
            reset_trackbars(control_window)
        elif key == ord('s'):
            save_parameters()

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

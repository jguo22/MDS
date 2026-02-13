"""
Detect the color of a can being picked up by comparing mask intersections
with a target rectangle in the gripper area.
"""
import numpy as np
from shapely.geometry import Polygon
from typing import Optional

from colors import CAN_CLASS_NAMES
from vision.mask_utils import yoloMaskToBinary, maskToConvexHull

# Rectangle defining the gripper pickup area in pixel coordinates (u, v)
PICKED_RECT = np.array([[295, 480], [295, 240], [540, 240], [540, 480]])


def get_intersection_area(
    convex_hull: Optional[np.ndarray],
    rectangle: np.ndarray
) -> float:
    """
    Calculate intersection area between a convex hull and rectangle.

    Args:
        convex_hull: Convex hull vertices in pixel (u, v) coordinates,
            as numpy array with shape (N, 1, 2) or (N, 2)
        rectangle: Rectangle vertices as numpy array with shape (4, 2)

    Returns:
        float: Intersection area in pixels, or 0.0 if no intersection
    """
    if convex_hull is None:
        return 0.0

    try:
        # Handle both (N, 1, 2) and (N, 2) shapes
        if convex_hull.ndim == 3:
            convex_hull = convex_hull.reshape(-1, 2)

        if convex_hull.shape[0] < 3:
            return 0.0

        hull_poly = Polygon(convex_hull)
        rect_poly = Polygon(rectangle)

        if not hull_poly.is_valid or not rect_poly.is_valid:
            return 0.0

        intersection = hull_poly.intersection(rect_poly)
        return intersection.area

    except Exception:
        return 0.0


def getPickedUpCanColor(result, image) -> Optional[str]:
    """
    Determine the color of a can being picked up by the gripper.

    Compares all can masks with the PICKED_RECT rectangle and returns
    the class name of the mask with the greatest intersection area.

    Args:
        result: YOLO segmentation model result
        image: Original BGR image (used for mask processing)

    Returns:
        str: Class name of the can with greatest intersection (e.g., 'Green Can'),
            or None if no can mask intersects with the gripper area
    """
    if result.masks is None:
        return None

    best_class_name = None
    best_intersection_area = 0.0

    for i, mask_orig in enumerate(result.masks):
        # Get class name for this detection
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]

        # Only process can detections
        if class_name not in CAN_CLASS_NAMES:
            continue

        # Convert YOLO mask to binary
        binary_mask = yoloMaskToBinary(mask_orig, image)

        # Get convex hull of mask
        hull_uv = maskToConvexHull(binary_mask)
        if hull_uv is None or len(hull_uv) == 0:
            continue

        # Calculate intersection area with gripper rectangle
        intersection_area = get_intersection_area(hull_uv, PICKED_RECT)

        # Track the mask with greatest intersection
        if intersection_area > best_intersection_area:
            best_intersection_area = intersection_area
            best_class_name = class_name

    return best_class_name


def main():
    """Example usage of getPickedUpCanColor."""
    import cv2
    from vision.segment import segmentImage

    # Load a test image
    image_path = "green.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
        return

    print(f"Loaded image: {image.shape}")

    # Run YOLO segmentation
    result = segmentImage(image)

    if result.masks is None:
        print("No masks detected")
        return

    print(f"Detected {len(result.masks)} masks")

    # Get the color of the can in the gripper area
    picked_color = getPickedUpCanColor(result, image)

    if picked_color is not None:
        print(f"Picked up can color: {picked_color}")
    else:
        print("No can detected in gripper area")

    # Visualize the gripper rectangle on the image
    cv2.polylines(image, [PICKED_RECT], isClosed=True,
                  color=(0, 255, 255), thickness=2)

    # Show result
    cv2.imshow("Gripper Area Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

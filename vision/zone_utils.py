"""
Zone detection and quadrilateral utilities.
"""
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
from typing import Tuple, Optional

from colors import GOLDEN_ZONE, ZONE_CLASS_NAMES
from config import BIG_ZONE_SIDE_LENGTH, SMALL_ZONE_SIDE_LENGTH

from .pixelTo3D import transform_uv_to_xy, H_TOP, H_BOTTOM
from .mask_utils import maskToConvexHull
from .relativeCoordinates import world_to_pixel
from spatialmath import SE2


def getZones(result, image, is_top=True, epsilon_factor=0.02):
    """
    Extract zones from YOLO result using mask.xy polygons and convert to world coordinates.

    Args:
        result: YOLO result object with masks
        image: Original image (for reference, not used for coordinates)
        is_top: True for top camera, False for bottom camera
        epsilon_factor: Polygon simplification factor (default: 0.02)

    Returns:
        Tuple of (zones, class_names, confidences):
        - zones: List of numpy arrays, each with shape (N, 2) representing polygon vertices in world coordinates (mm)
        - class_names: List of zone class names (e.g., 'Green Zone')
        - confidences: List of detection confidence values
    """
    zones = []
    class_names = []
    confidences = []

    if result.masks is None or len(result.masks) == 0:
        return zones, class_names, confidences

    for i, mask in enumerate(result.masks):
        # Get class info
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]
        confidence = float(result.boxes.conf[i])

        # Skip if not a zone
        if class_name not in ZONE_CLASS_NAMES:
            continue

        # Get polygon from mask.xy
        if not hasattr(mask, 'xy') or len(mask.xy) == 0:
            continue

        # Extract polygon in pixel coordinates
        poly_uv = mask.xy[0]
        if hasattr(poly_uv, 'cpu'):
            poly_uv = poly_uv.cpu().numpy()
        else:
            poly_uv = np.array(poly_uv)

        if len(poly_uv) < 3:
            continue

        # Simplify polygon
        poly_uv_simplified = simplify_polygon(poly_uv, epsilon_factor)

        # Transform to world coordinates
        poly_xy = []
        for point in poly_uv_simplified:
            u, v = point[0], point[1]
            xy = transform_uv_to_xy(u, v, is_top)
            if xy is not None:
                poly_xy.append(xy)

        if len(poly_xy) < 3:
            continue

        poly_xy = np.array(poly_xy)

        zones.append(poly_xy)
        class_names.append(class_name)
        confidences.append(confidence)

        # print(f"{class_name}: {len(poly_xy)} vertices (confidence: {confidence:.2f})")

    return zones, class_names, confidences


def simplify_polygon(polygon, epsilon_factor=0.02):
    """Simplify polygon using Douglas-Peucker algorithm.

    Returns numpy array with float64 dtype for JSON serialization compatibility.
    """
    perimeter = cv.arcLength(polygon.astype(np.float32), closed=True)
    epsilon = epsilon_factor * perimeter
    approx = cv.approxPolyDP(polygon.astype(np.float32), epsilon, closed=True)
    result = approx.reshape(-1, 2)
    return result.astype(np.float64)


def getPolygonCenter(polygon) -> Tuple[float, float]:
    """
    Calculates the center point (centroid) of a polygon.

    Args:
        polygon: Polygon vertices as numpy array with shape (N, 1, 2) or (N, 2)
              Format: [[x1, y1], [x2, y2], ..., [xN, yN]]
              where N is the number of vertices

    Returns:
        tuple: (center_x, center_y) as floats representing the centroid
    """
    # Reshape from (N, 1, 2) to (N, 2) for easier processing
    points = polygon.reshape(-1, 2)

    # Calculate mean of all x and y coordinates
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])

    return (center_x, center_y)


def isPointInPoly(point, polygon):
    """
    Check if a point is inside a polygon.

    Uses OpenCV's pointPolygonTest for accurate and efficient polygon containment check.

    Args:
        point: Point coordinates as tuple (x, y) in same units as polygon
        polygon: Polygon vertices as numpy array with shape (N, 1, 2) or (N, 2)
              Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        bool: True if point is inside polygon (or on boundary), False otherwise
    """
    # Convert to format expected by pointPolygonTest: (N, 1, 2) with float32
    contour = polygon.reshape(-1, 1, 2).astype(np.float32)

    # pointPolygonTest returns:
    # > 0: point is inside
    # = 0: point is on edge
    # < 0: point is outside
    result = cv.pointPolygonTest(contour, point, measureDist=False)

    return result >= 0


def doPolygonsIntersect(poly1, poly2) -> bool:
    """
    Check if two polygons intersect.

    Uses Shapely's intersection test to determine if two polygons overlap.
    Handles various input formats including numpy arrays and existing Shapely Polygons.

    Args:
        poly1: First polygon as:
               - Numpy array with shape (N, 2) or (N, 1, 2)
               - Shapely Polygon object
        poly2: Second polygon (same format options as poly1)

    Returns:
        bool: True if polygons intersect (share any area or touch), False otherwise.
              Returns False if either input is None or invalid.

    Examples:
        >>> import numpy as np
        >>> p1 = np.array([[0,0], [100,0], [100,100], [0,100]])
        >>> p2 = np.array([[50,50], [150,50], [150,150], [50,150]])
        >>> doPolygonsIntersect(p1, p2)
        True
        >>> p3 = np.array([[200,200], [300,200], [300,300], [200,300]])
        >>> doPolygonsIntersect(p1, p3)
        False
    """
    try:
        # Convert poly1 to Shapely Polygon
        if isinstance(poly1, Polygon):
            shapely_poly1 = poly1
        elif isinstance(poly1, np.ndarray):
            # Handle both (N, 1, 2) and (N, 2) shapes
            if poly1.ndim == 3:
                poly1 = poly1.reshape(-1, 2)
            shapely_poly1 = Polygon(poly1)
        else:
            # Invalid input type
            return False

        # Convert poly2 to Shapely Polygon
        if isinstance(poly2, Polygon):
            shapely_poly2 = poly2
        elif isinstance(poly2, np.ndarray):
            # Handle both (N, 1, 2) and (N, 2) shapes
            if poly2.ndim == 3:
                poly2 = poly2.reshape(-1, 2)
            shapely_poly2 = Polygon(poly2)
        else:
            # Invalid input type
            return False

        # Check if polygons are valid
        if not shapely_poly1.is_valid or not shapely_poly2.is_valid:
            return False

        # Check for intersection
        return shapely_poly1.intersects(shapely_poly2)

    except (ValueError, TypeError, AttributeError):
        # Return False for any errors (invalid inputs, empty arrays, etc.)
        return False


def annotate_poly(image, polygon, color=(0, 0, 255)):
    """
    Draws polygon on image with corner points.

    Args:
        image: Image to annotate (numpy array)
              Shape: (H, W, 3) for BGR color image
        polygon: Polygon contour as numpy array, shape (N, 2) or (N, 1, 2)
        color: Color for polygon (BGR tuple), default red (0, 0, 255)

    Returns:
        Annotated image
    """
    # Reshape to (N, 1, 2) for drawContours
    polygon_reshaped = polygon.reshape(-1, 1, 2).astype(np.int32) if polygon.ndim == 2 else polygon.astype(np.int32)

    # Draw the polygon
    cv.drawContours(image, [polygon_reshaped], 0, color, 1)

    # Draw corner points
    points = polygon.reshape(-1, 2)
    for point in points:
        cv.circle(image, tuple(point.astype(np.int32)), 5, color, -1)

    return image


def visualize_convex_hulls(
    image: np.ndarray,
    result,
    color: tuple = (255, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Visualize convex hulls of all segmentation masks on the image.

    Args:
        image: Image to draw on (BGR format, numpy array)
        result: YOLO result object with masks
        color: BGR color tuple for drawing hulls (default: cyan)
        thickness: Line thickness for hull outline (default: 2)

    Returns:
        Image with convex hulls drawn
    """
    output_image = image.copy()

    if result.masks is None:
        return output_image

    for i, mask_orig in enumerate(result.masks):
        # Get class name
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]

        # Convert mask to grayscale image
        mask_array = mask_orig.data[0].cpu().numpy()
        mask_uint8 = (mask_array * 255).astype(np.uint8)

        # Resize mask to match original image size
        # Use INTER_NEAREST for binary masks to avoid interpolation artifacts
        # and shifts
        mask_resized = cv.resize(
            mask_uint8,
            (image.shape[1],
             image.shape[0]),
            interpolation=cv.INTER_NEAREST)

        # Get convex hull
        try:
            hull = maskToConvexHull(mask_resized)
            if hull is None or len(hull) == 0:
                continue
        except Exception:
            continue

        # Draw convex hull
        cv.drawContours(output_image, [hull], 0, color, thickness)

        # Optionally draw corner points
        hull_reshaped = hull.reshape(-1, 2) if hull.ndim == 3 else hull
        for point in hull_reshaped:
            cv.circle(
                output_image, (int(point[0]), int(point[1])), 3, color, -1)
    return output_image


def visualize_xy_locations(
    image: np.ndarray,
    xy_points: list,
    robot_pose: SE2,
    is_top: bool = True,
    color: tuple = (0, 255, 0),
    radius: int = 5,
    thickness: int = -1,
    labels: Optional[list] = None
) -> np.ndarray:
    """
    Visualize world coordinate (x, y) locations on an image by projecting them to pixels.

    Takes a list of points in world coordinates (mm) and draws them on the image
    after converting to pixel coordinates using the camera's homography matrix.

    Args:
        image: Image to draw on (BGR format, numpy array)
        xy_points: List of (x, y) tuples in world coordinates (mm)
                  For camera-relative coordinates:
                  - x: forward distance from camera (positive = in front)
                  - y: lateral distance from camera (positive = left)
        robot_pose: Robot's current pose (SE2) - used if points are in world frame
        is_top: True for top camera, False for bottom camera
        color: BGR color tuple for drawing points (default: green)
        radius: Radius of circles to draw (default: 5)
        thickness: Thickness of circle outline (-1 for filled, default: -1)
        labels: Optional list of labels to display next to each point

    Returns:
        Image with xy locations visualized as circles

    Example:
        >>> image = cv2.imread("frame.jpg")
        >>> can_locations = [(500, 100), (800, -200), (1200, 0)]  # in mm
        >>> robot_pose = SE2(0, 0, 0)
        >>> viz_image = visualize_xy_locations(image, can_locations, robot_pose, is_top=True)
        >>> cv2.imshow("Cans", viz_image)
    """
    # Make a copy to avoid modifying the original
    output_image = image.copy()

    # Select the appropriate homography matrix
    h_matrix = H_TOP if is_top else H_BOTTOM

    # Draw each point
    for i, xy_point in enumerate(xy_points):
        if xy_point is None:
            continue

        # Convert world coordinates to pixel coordinates
        pixel_coords = world_to_pixel(xy_point, h_matrix)

        if pixel_coords is None:
            # Point cannot be projected (behind camera or invalid)
            continue

        u, v = pixel_coords

        # Check if point is within image bounds
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            # Draw circle at the location
            cv.circle(output_image, (u, v), radius, color, thickness)

            # Draw label if provided
            if labels is not None and i < len(labels):
                label = str(labels[i])
                # Put text slightly offset from the circle
                text_pos = (u + radius + 5, v + 5)
                cv.putText(
                    output_image,
                    label,
                    text_pos,
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv.LINE_AA
                )

    return output_image

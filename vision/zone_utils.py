"""
Zone detection and quadrilateral utilities.
"""
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon

from colors import GOLDEN_ZONE, ZONE_CLASS_NAMES
from config import BIG_ZONE_SIDE_LENGTH, SMALL_ZONE_SIDE_LENGTH

from .pixelTo3D import transform_uv_to_xy
from .mask_utils import maskToConvexHull


def getSquareCenter(square):
    """
    Calculates the center point of a square.

    Args:
        square: Square vertices as numpy array with shape (4, 1, 2) or (4, 2)
              Format: [[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]]

    Returns:
        tuple: (center_x, center_y) as integers
    """
    # Reshape from (4, 1, 2) to (4, 2) for easier processing
    points = square.reshape(4, 2)

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
        polygon: Polygon contour (N, 1, 2) numpy array
        color: Color for polygon (BGR tuple), default red (0, 0, 255)

    Returns:
        Annotated image
    """
    # Draw the polygon
    cv.drawContours(image, [polygon], 0, color, 1)

    # Draw corner points
    for point in polygon:
        cv.circle(image, tuple(point[0]), 5, color, -1)

    return image


def approximateConvexHullWithSquare(convexHull, side_length):
    """
    Approximates a region defined by a contour with a rotated square.

    The square is centered at the geometric centroid and has the specified side length.
    The square's orientation is determined by testing multiple angles and choosing the
    one with maximum overlap (intersection area) with the original contour.

    Args:
        contour: Contour/polygon vertices as numpy array with shape (N, 1, 2) or (N, 2)
                 where N is the number of vertices. Can be in any coordinate system (pixels, mm, etc.)
        side_length: Side length of the square in the same units as the contour
        num_angles: Number of angles to test (default: 36, tests every 5 degrees)

    Returns:
        tuple: (square, iou)
            - square: numpy array of shape (4, 2) representing the square's corners
            - iou: float representing intersection area / hull area (0.0 to 1.0)
        Returns (None, 0.0) if contour is empty, invalid, or if shapely operations fail.
    """
    if convexHull is None or len(convexHull) == 0:
        return None, 0.0

    try:
        # Handle both (N, 1, 2) and (N, 2) shapes
        if convexHull.ndim == 3:
            convexHull = convexHull.reshape(-1, 2)

        # Create shapely Polygon to compute geometric centroid
        # and for overlap calculation
        polygon = Polygon(convexHull)

        # Skip invalid polygons instead of trying to fix them
        # (buffer(0) can change shape significantly)
        if not polygon.is_valid:
            return None, 0.0

        centroid = polygon.centroid
        center_x = centroid.x
        center_y = centroid.y

        # Calculate half side length
        half_side = side_length / 2.0

        # Create square vertices in local coordinates (centered at origin)
        local_square = np.array([
            [-half_side, -half_side],
            [half_side, -half_side],
            [half_side, half_side],
            [-half_side, half_side],
        ])

        # Test multiple angles and find the one with best overlap
        best_angle = 0
        best_overlap = 0
        # 0 to 90 degrees (squares have 4-fold symmetry)
        num_angles = 36
        angles_to_test = np.linspace(0, np.pi / 2, num_angles)

        for angle_rad in angles_to_test:
            # Create rotation matrix
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_angle, -sin_angle],
                [sin_angle, cos_angle]
            ])

            # Rotate and translate the square
            rotated_square = local_square @ rotation_matrix.T
            square_vertices = rotated_square + np.array([center_x, center_y])

            square_polygon = Polygon(square_vertices)

            # Calculate intersection area
            intersection = polygon.intersection(square_polygon)
            overlap_area = intersection.area

            # Track the best angle
            if overlap_area > best_overlap:
                best_overlap = overlap_area
                best_angle = angle_rad

        # Create the final square with the best angle
        cos_angle = np.cos(best_angle)
        sin_angle = np.sin(best_angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        rotated_square = local_square @ rotation_matrix.T
        square = rotated_square + np.array([center_x, center_y])

        # Calculate IoU between convex hull and square
        square_polygon = Polygon(square)
        iou = best_overlap / polygon.area if polygon.area > 0 else 0.0

        return square, iou

    except Exception:
        # Return None if any shapely or numpy operations fail
        return None, 0.0


def getZones(result, image, is_top=True):
    """
    Extracts zones from YOLO results and approximates each as a square.

    Processes zones directly without using getZones(), implementing the full pipeline
    from YOLO results to square approximations in world coordinates.

    Args:
        result: YOLO result object from inference
        image: Original BGR image (used for fixSegmentation)
        side_length: Side length of the approximating squares in mm

    Returns:
        tuple: (squares_xy, class_names, confidence_scores)
            - squares_xy: List of numpy arrays (4, 2) with xy coordinates in mm
                         Each array represents a square centered at the zone's centroid
            - class_names: List of strings with class names for each square
            - confidence_scores: List of floats representing IoU between hull and square in xy coordinates
    """
    squares = []
    class_names = []
    confidence_scores = []

    if result.masks is None:
        return squares, class_names, confidence_scores

    for i, mask_orig in enumerate(result.masks):
        # Get class name for this detection
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]

        # if its not a zone, continue to next mask
        if class_name not in ZONE_CLASS_NAMES:
            continue

        # Convert mask to grayscale image
        mask_array = mask_orig.data[0].cpu().numpy()
        mask_uint8 = (mask_array * 255).astype(np.uint8)

        # Resize mask to match original image size
        mask_resized = cv.resize(
            mask_uint8, (image.shape[1], image.shape[0]))

        # Get convex hull in pixel coordinates (N, 1, 2)
        try:
            hull_uv = maskToConvexHull(mask_resized)
            if hull_uv is None or len(hull_uv) == 0:
                continue
        except Exception:
            continue

        # Reshape from (N, 1, 2) to (N, 2) for easier iteration
        hull_uv_reshaped = hull_uv.reshape(-1, 2)

        # Transform hull points from pixel to world coordinates
        hull_xy = []
        valid_hull = True
        for point in hull_uv_reshaped:
            u, v = point[0], point[1]
            xy = transform_uv_to_xy(u, v, is_top)
            if xy is None:
                valid_hull = False
                break
            hull_xy.append(xy)

        # move on to next mask if this isn't a valid hull
        if not valid_hull:
            continue

        hull_xy = np.array(hull_xy)

        # Approximate with square based on zone type
        if class_name == ZONE_CLASS_NAMES[GOLDEN_ZONE]:
            square, iou = approximateConvexHullWithSquare(
                hull_xy, SMALL_ZONE_SIDE_LENGTH)
        else:
            square, iou = approximateConvexHullWithSquare(
                hull_xy, BIG_ZONE_SIDE_LENGTH)

        if square is not None:
            squares.append(square)
            class_names.append(class_name)
            confidence_scores.append(iou)

    return squares, class_names, confidence_scores

"""
Zone detection and quadrilateral utilities.
"""
import cv2 as cv
import numpy as np
from pixelTo3D import transform_uv_to_xy
from mask_utils import fixSegmentation, calculateQuadFromMask


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


def annotate_poly(image, polygon, color=(0, 0, 255)):
    """
    Draws polygon on image with corner points.

    Args:
        image: Image to annotate (numpy array)
              Shape: (H, W, 3) for BGR color image
        polygon: Polygon contour (N, 1, 2) numpy array
        color: Color for quadrilateral (BGR tuple), default red (0, 0, 255)

    Returns:
        Annotated image
    """
    # Draw the polygon
    cv.drawContours(image, [polygon], 0, color, 1)

    # Draw corner points
    for point in polygon:
        cv.circle(image, tuple(point[0]), 5, color, -1)

    return image


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


def getZones(result, image):
    """
    Extracts quadrilaterals from YOLO results and transforms to xy coordinates.

    Args:
        result: YOLO result object from inference
        image: Original BGR image (used for fixSegmentation)

    Returns:
        tuple: (quads_xy, class_names)
            - quads_xy: List of numpy arrays (4, 2) with xy coordinates in mm
            - class_names: List of strings with class names
    """

    # Get quadrilaterals in pixel coordinates
    quads_pixel, class_names = getQuadrilateralsAndClasses(result, image)

    if len(quads_pixel) == 0:
        return [], []

    # Transform each vertex from pixel to xy coordinates
    transformed_quads = []

    for quad in quads_pixel:
        # Reshape from (4, 1, 2) to (4, 2)
        vertices = quad.reshape(4, 2)

        # Transform each vertex
        transformed_vertices = []
        isValidQuad = True
        for vertex in vertices:
            u, v = vertex[0], vertex[1]  # pixel coordinates
            xy = transform_uv_to_xy(u, v)  # ground plane coordinates (mm)
            print(xy)
            if xy is None:
                isValidQuad = False
                break
            else:
                x, y = xy
                transformed_vertices.append([x, y])
        if not isValidQuad:
            continue

        transformed_quad = np.array(transformed_vertices)
        transformed_quads.append(transformed_quad)

    return transformed_quads, class_names

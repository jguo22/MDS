import numpy as np
from yolo.segment import getQuadrilateralsAndClasses


def getZones(result, image):
    """
    Extracts quadrilaterals from YOLO results and transforms to xy coordinates,
    sorted by distance from robot.

    Args:
        result: YOLO result object from inference
        image: Original BGR image (used for fixSegmentation)

    Returns:
        tuple: (quads_xy, class_names)
            - quads_xy: List of numpy arrays (4, 2) with xy coordinates in mm
            - class_names: List of strings with class names
            Both arrays are sorted by distance from robot (closest first)
    """
    from pixelTo3D import transform_uv_to_xy

    # Get quadrilaterals in pixel coordinates
    quads_pixel, class_names = getQuadrilateralsAndClasses(result, image)

    if len(quads_pixel) == 0:
        return [], []

    # Transform each vertex from pixel to xy coordinates
    transformed_quads = []
    distances = []

    for quad in quads_pixel:
        # Reshape from (4, 1, 2) to (4, 2)
        vertices = quad.reshape(4, 2)

        # Transform each vertex
        transformed_vertices = []
        for vertex in vertices:
            u, v = vertex[0], vertex[1]  # pixel coordinates
            x, y = transform_uv_to_xy(u, v)  # ground plane coordinates (mm)
            transformed_vertices.append([x, y])

        transformed_quad = np.array(transformed_vertices)
        transformed_quads.append(transformed_quad)

        # Calculate center and distance for sorting
        center_x = np.mean(transformed_quad[:, 0])
        center_y = np.mean(transformed_quad[:, 1])
        distance = np.sqrt(center_x**2 + center_y**2)
        distances.append(distance)

    # Sort by distance (closest first)
    sorted_indices = np.argsort(distances)
    quads_xy = [transformed_quads[i] for i in sorted_indices]
    class_names_sorted = [class_names[i] for i in sorted_indices]

    return quads_xy, class_names_sorted

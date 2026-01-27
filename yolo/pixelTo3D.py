import numpy as np

H_TOP = np.array([[-6.09741811e-01, - 2.09501156e-02, 1.57159029e+02],
                  [2.85708157e-02, 9.39073240e-03, - 5.18950601e+02],
                  [6.09393965e-04, - 7.78799171e-03, 1.00000000e+00]])
H_BOTTOM = np.array([[-6.09741811e-01, - 2.09501156e-02, 1.57159029e+02],
                     [2.85708157e-02, 9.39073240e-03, - 5.18950601e+02],
                     [6.09393965e-04, - 7.78799171e-03, 1.00000000e+00]])


def transform_uv_to_xy(u, v, is_top=True):
    """
    u and v are pixel coordinates.
    The top left pixel is the origin, u axis increases to right, and v axis
    increases down.
    Returns a normal non-np 1x2 matrix of xy displacement vector from the
    camera to the point on the ground plane.
    Camera points along positive x axis and y axis increases to the left of
    the camera.
    Units are in whichever unit h was calculated in.
    h is the homography matrix
    """
    homogeneous_point = np.array([[u], [v], [1]])
    if is_top:
        h_matrix = H_TOP
    else:
        h_matrix = H_BOTTOM
    xy = np.dot(h_matrix, homogeneous_point)
    scaling_factor = 1.0 / xy[2, 0]
    homogeneous_xy = xy * scaling_factor
    x = homogeneous_xy[0, 0]
    y = homogeneous_xy[1, 0]

    if y < 0:
        return None

    return x, y


def transform_contour_to_xy(contour, is_top=True):
    """
    Transform an entire contour from pixel coordinates to world coordinates.

    Args:
        contour: OpenCV contour array of shape (N, 1, 2) or (N, 2) where each
                 point is [u, v] in pixel coordinates

    Returns:
        numpy array of shape (M, 2) where each point is [x, y] in world coordinates (mm).
        Points that are behind the camera (y < 0) are filtered out, so M <= N.
        Returns None if no valid points remain after transformation.
    """
    # Handle both (N, 1, 2) and (N, 2) shapes
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)

    transformed_points = []

    for point in contour:
        u, v = point[0], point[1]
        result = transform_uv_to_xy(u, v, is_top)

        if result is not None:
            x, y = result
            transformed_points.append([x, y])

    if len(transformed_points) == 0:
        return None

    return np.array(transformed_points)

from vision.relativeCoordinates import world_to_pixel
from vision.relativeCoordinates import world_to_relative
import numpy as np
from config import FRAME_WIDTH, FRAME_HEIGHT

H_TOP = np.array([[-3.26587760e-01, -1.19235902e+00, 4.41202786e+03],
                  [-4.62826170e+00, -5.02535675e-03, 1.70630658e+03],
                  [-3.14460034e-04, 8.10449367e-03, 1.00000000e+00]])
H_BOTTOM = np.array([[9.03915217e-03, -6.24788513e-01, 4.39028497e+02],
                     [-7.09578335e-01, -1.23654833e-02, 2.99076080e+02],
                     [-2.24196211e-05, 6.01813413e-04, 1.00000000e+00]])


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

    if x < 0:
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
        Points that are behind the camera (x < 0) are filtered out, so M <= N.
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


def is_world_point_visible(
        world_x: float,
        world_y: float,
        is_top: bool) -> bool:
    """
    Check if a world point is visible in the camera's field of view.

    Args:
        world_x: x coordinate in world frame (mm)
        world_y: y coordinate in world frame (mm)
        is_top: True for top camera, False for bottom camera

    Returns:
        True if the point is visible in the specified camera's FOV
    """

    # Convert world coordinates to robot-relative coordinates
    camera_relative = world_to_relative(
        (world_x, world_y), self.robot_pose)

    # Points behind the camera cannot be visible
    if camera_relative[0] < 0:
        return False

    # Get the appropriate homography matrix
    h_matrix = H_TOP if is_top else H_BOTTOM

    # Try to project to pixel coordinates
    pixel_coords = world_to_pixel(camera_relative, h_matrix)
    if pixel_coords is None:
        return False

    # Check if pixel coordinates are within frame bounds
    u, v = pixel_coords
    return 0 <= u < FRAME_WIDTH and 0 <= v < FRAME_HEIGHT

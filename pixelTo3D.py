import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R


CAMERA_MATRIX = np.array([[900.83135648, 0, 319.13723878],
                          [0, 905.17695622, 236.54418761],
                          [0, 0, 1]])
DISTORTION = np.array([[9.62758290e-02, 7.15871128e-01,
                        3.69387355e-03, 1.18130977e-02, -6.76390055e+00]])
# HEIGHT_MM = 99.06
# ANGLE_MATRIX = R.from_euler('x', -1).as_matrix()


def undistort_pixel(pixel_x, pixel_y):
    pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    undistorted = cv.undistortPoints(
        pixel, camera_matrix, distortion, None, camera_matrix)
    pixel_undist = undistorted[0, 0]
    return pixel_undist


def pixel_to_camera_coords(
    pixel_x,  # 0 to 640
    pixel_y,  # 0 to 480
):
    # TODO: FIX THIS
    """
    Convert a pixel coordinate to 3D camera coordinates.
    NOTE: USING 640x480

    Args:
        pixel_x: X coordinate of the pixel (column, horizontal position)
        pixel_y: Y coordinate of the pixel (row, vertical position)
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        depth: Optional depth value. If None, returns normalized ray direction.
                   If provided, returns 3D position at that depth.

    Returns:
        tuple: (ray_direction, point_3d_camera)
            - ray_direction: 3D unit direction vector from camera center through pixel
            - point_3d_camera: 3D position in camera coordinates (only if depth provided)
    """
    # Step 1: Undistort the pixel coordinate
    pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    undistorted = cv.undistortPoints(
        pixel, CAMERA_MATRIX, DISTORTION, None, CAMERA_MATRIX)
    pixel_undist = undistorted[0, 0]
    print("pixel undistorted is")
    print(pixel_undist)

    ray_direction = CAMERA_MATRIX @ np.array(
        [pixel_undist[0], pixel_undist[1], 1]).T

    # Step 4: If depth provided, calculate actual 3D position
    return ray_direction


def pixel_to_robot_horizontal(pixel_x, pixel_y):
    ray_direction = pixel_to_camera_coords(pixel_x, pixel_y)

    ray_direction = ANGLE_MATRIX @ ray_direction

    x = ray_direction[0]
    y = ray_direction[1]
    z = ray_direction[2]

    x_scaled = x / z * HEIGHT_MM
    y_scaled = y / z * HEIGHT_MM
    return x_scaled, y_scaled


h = np.array([[-6.09741811e-01, - 2.09501156e-02, 1.57159029e+02],
              [2.85708157e-02, 9.39073240e-03, - 5.18950601e+02],
              [6.09393965e-04, - 7.78799171e-03, 1.00000000e+00]])


def transform_uv_to_xy(u, v):
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
    xy = np.dot(h, homogeneous_point)
    scaling_factor = 1.0 / xy[2, 0]
    homogeneous_xy = xy * scaling_factor
    x = homogeneous_xy[0, 0]
    y = homogeneous_xy[1, 0]
    return x, y

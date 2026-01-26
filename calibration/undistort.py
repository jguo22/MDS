import cv2 as cv
import numpy as np

# FRAME SIZE 864 by 448

# camera with case cover
DISTORTION_TOP = np.array([[0.05532577, 0.24960091,
                            0.00291173, -0.00237441, -2.0304583]])
MATRIX_TOP = np.array([[855.63090112, 0., 426.85535747],
                       [0., 852.87147114, 241.977431],
                       [0., 0., 1.]])

# camera without case
DISTORTION_BOTTOM = np.array([[4.96212321e-02, 5.63787443e-01,
                               -1.32777756e-03, 7.70730637e-03, -3.61891305e+00]])
MATRIX_BOTTOM = np.array([[877.10363722, 0., 451.85045204],
                          [0., 875.16155798, 233.26825088],
                          [0., 0., 1.]])


def undistort(img_points: np.ndarray, is_top: bool):
    # img_points is a list of [u,v]
    if is_top:
        distortion = DISTORTION_TOP
        mtx = MATRIX_TOP
    else:
        distortion = DISTORTION_BOTTOM
        mtx = MATRIX_BOTTOM

    imgpoints_undistorted = cv.undistortPoints(
        img_points, mtx, distortion, None, mtx)

    return imgpoints_undistorted


def undistortImage(img, is_top: bool):
    if is_top:
        distortion = DISTORTION_TOP
        mtx = MATRIX_TOP
    else:
        distortion = DISTORTION_BOTTOM
        mtx = MATRIX_BOTTOM

    # refine camera matrix
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, distortion, (w, h), 1, (w, h))

# undistort
    dst = cv.undistort(img, mtx, distortion, None, newcameramtx)

# crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    return dst

import numpy as np
import cv2 as cv
import glob

# Grid dimensions for chessboard calibration
GRID_WIDTH = 8  # Number of inner corners along width
GRID_HEIGHT = 11  # Number of inner corners along height

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((GRID_WIDTH * GRID_HEIGHT, 3), np.float32)
objp[:, :2] = np.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('images/*.jpg')

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Warning: Could not read image {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(
        gray, (GRID_HEIGHT, GRID_WIDTH), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (GRID_HEIGHT, GRID_WIDTH), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(5)

cv.destroyAllWindows()

print(f"number of good images: {len(imgpoints)}")

# calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

for image_index in range(len(images)):
    fname = images[image_index]
    img = cv.imread(fname)
# refine camera matrix
    img = cv.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    print("asdfsd")
    print(newcameramtx)
    print(roi)

# undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite(f'calibration_results/calibresult{image_index}.png', dst)

# reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(objpoints)))

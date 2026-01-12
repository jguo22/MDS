import numpy as np
import cv2 as cv
import glob

# Grid dimensions for chessboard calibration
GRID_WIDTH = 6  # Number of inner corners along width
GRID_HEIGHT = 7  # Number of inner corners along height

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((GRID_WIDTH * GRID_HEIGHT, 3), np.float32)
objp[:, :2] = np.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('*.jpg')

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
        cv.waitKey(500)

cv.destroyAllWindows()

# Check if we found any valid calibration images
if len(objpoints) == 0:
    print("Error: No valid calibration images found!")
    exit(1)

# Get image size from first image in the list
first_img = cv.imread(images[0])
if first_img is None:
    print("Error: Could not read first image for calibration")
    exit(1)

img_size = (first_img.shape[1], first_img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, img_size, None, None)


img = cv.imread('left12.jpg')
if img is None:
    print("Error: Could not read image 'left12.jpg'")
    exit(1)

h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))


# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('calibresult.png', dst)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))

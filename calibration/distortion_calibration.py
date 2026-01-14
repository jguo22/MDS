import numpy as np
import cv2 as cv
import glob
from pixelTo3D import pixel_to_camera_coords

# Grid dimensions for chessboard calibration
GRID_ROWS = 11  # Number of inner corners along height
GRID_COLUMNS = 8  # Number of inner corners along width

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((GRID_ROWS * GRID_COLUMNS, 3), np.float32)
objp[:, :2] = np.mgrid[0:GRID_ROWS, 0:GRID_COLUMNS].T.reshape(-1, 2)

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
        gray, (GRID_ROWS, GRID_COLUMNS), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (GRID_ROWS, GRID_COLUMNS), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

print(f"number of good images: {len(imgpoints)}")

# calibration
ret, mtx, distortion, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print(f'ret: {ret}')
print(f'mtx: {mtx}')
print(f'distortion: {distortion}')

for image_index in range(len(images)):
    fname = images[image_index]
    img = cv.imread(fname)
# refine camera matrix
    img = cv.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, distortion, (w, h), 1, (w, h))

# undistort
    dst = cv.undistort(img, mtx, distortion, None, newcameramtx)

# crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite(f'calibration_results/calibresult{image_index}.png', dst)

# reprojection error using reverse projection (2D -> 3D on z=0 plane)
mean_error = 0
for i in range(len(objpoints)):
    # Undistort the detected 2D image points
    imgpoints_undistorted = cv.undistortPoints(
        imgpoints[i], mtx, distortion, None, mtx)

    # Convert rotation vector to rotation matrix
    R, _ = cv.Rodrigues(rvecs[i])
    t = tvecs[i].flatten()  # Ensure 1D array

    # Camera position in world coordinates
    camera_pos = (-R.T @ tvecs[i]).flatten()  # Ensure 1D array

    # Project each 2D point to 3D on z=0 plane
    objpoints_reconstructed = []
    for point_2d in imgpoints_undistorted:
        # Normalized image coordinates
        x_norm = (point_2d[0][0] - mtx[0, 2]) / mtx[0, 0]
        y_norm = (point_2d[0][1] - mtx[1, 2]) / mtx[1, 1]

        # Ray direction in camera coordinates
        ray_cam = np.array([x_norm, y_norm, 1.0])

        # Transform ray to world coordinates
        ray_world = R.T @ ray_cam

        # Find intersection with z=0 plane
        # camera_pos + t * ray_world = [x, y, 0]
        # Solve for t: camera_pos[2] + t * ray_world[2] = 0
        t_intersect = -camera_pos[2] / ray_world[2]

        # Calculate 3D point on z=0 plane
        point_3d = camera_pos + t_intersect * ray_world
        objpoints_reconstructed.append(point_3d)  # Only x, y (z is 0)

    # Convert to numpy array with matching type and shape
    objpoints_reconstructed = np.array(
        objpoints_reconstructed, dtype=np.float32)

    # Compare reconstructed 3D points with original object points
    error = cv.norm(objpoints[i], objpoints_reconstructed,
                    cv.NORM_L2) / len(objpoints[i])
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))


# Example usage: Convert a pixel to 3D camera coordinates
print("\n--- Example: Pixel to 3D Camera Coordinates ---")

# Test with center pixel
pixel_x, pixel_y = 320, 240
print(f"Input pixel: ({pixel_x}, {pixel_y})")

# Get ray direction only
ray_dir = pixel_to_camera_coords(pixel_x, pixel_y, mtx, distortion)
print(f"\nRay direction in camera coordinates: {ray_dir}")
print(f"  [X={ray_dir[0]:.4f}, Y={ray_dir[1]:.4f}, Z={ray_dir[2]:.4f}]")

print("\nNote: Without depth info, you only get the ray direction.")
print("To find exact 3D position, you need:")
print("  1. Known depth (from depth sensor, stereo vision, etc.)")
print("  2. Known constraint (e.g., point lies on ground plane z=0)")
print("  3. Known object size and use pose estimation")

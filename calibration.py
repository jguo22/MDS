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
        # cv.imshow('img', img)
        # cv.waitKey(500)

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


def pixel_to_camera_coords(pixel_x, pixel_y, camera_matrix, dist_coeffs, depth=None):
    """
    Convert a pixel coordinate to 3D camera coordinates.

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
    undistorted = cv.undistortPoints(pixel, camera_matrix, dist_coeffs, None, camera_matrix)
    u_undist = undistorted[0, 0, 0]
    v_undist = undistorted[0, 0, 1]

    # Step 2: Extract camera intrinsic parameters
    fx = camera_matrix[0, 0]  # Focal length in x
    fy = camera_matrix[1, 1]  # Focal length in y
    cx = camera_matrix[0, 2]  # Principal point x (optical center)
    cy = camera_matrix[1, 2]  # Principal point y (optical center)

    # Step 3: Convert to normalized camera coordinates (ray direction)
    x_normalized = (u_undist - cx) / fx
    y_normalized = (v_undist - cy) / fy
    z_normalized = 1.0

    ray_direction = np.array([x_normalized, y_normalized, z_normalized])

    # Step 4: If depth provided, calculate actual 3D position
    if depth is not None:
        point_3d_camera = ray_direction * depth
        return ray_direction, point_3d_camera
    else:
        return ray_direction, None


# Example usage: Convert a pixel to 3D camera coordinates
print("\n--- Example: Pixel to 3D Camera Coordinates ---")

# Test with center pixel
pixel_x, pixel_y = 320.0, 240.0
print(f"Input pixel: ({pixel_x}, {pixel_y})")

# Get ray direction only
ray_dir, _ = pixel_to_camera_coords(pixel_x, pixel_y, mtx, distortion)
print(f"\nRay direction in camera coordinates: {ray_dir}")
print(f"  [X={ray_dir[0]:.4f}, Y={ray_dir[1]:.4f}, Z={ray_dir[2]:.4f}]")

# Get 3D position at specific depth
assumed_depth = 100.0
ray_dir, pos_3d = pixel_to_camera_coords(pixel_x, pixel_y, mtx, distortion, depth=assumed_depth)
print(f"\n3D position at depth={assumed_depth}:")
print(f"  [X={pos_3d[0]:.2f}, Y={pos_3d[1]:.2f}, Z={pos_3d[2]:.2f}]")

print("\nNote: Without depth info, you only get the ray direction.")
print("To find exact 3D position, you need:")
print("  1. Known depth (from depth sensor, stereo vision, etc.)")
print("  2. Known constraint (e.g., point lies on ground plane z=0)")
print("  3. Known object size and use pose estimation")

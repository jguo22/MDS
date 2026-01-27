import cv2
import numpy as np

IMAGE_WINDOW_NAME = "estimation"


def outputToArray(input: str) -> np.ndarray:
    lines = input.splitlines()
    points_2d = []
    for line in lines:
        commaIndex = line.find(',')
        firstColon = line.find(':')
        u = int(line[firstColon + 2:commaIndex])
        v = int(line[commaIndex + 5:])
        points_2d.append([u, v])
    return np.array(points_2d)


PTS_IMAGE_PLANE = outputToArray('''0 - u: 147, v: 261
1 - u: 362, v: 233
2 - u: 205, v: 177
3 - u: 33, v: 197
4 - u: 467, v: 144
5 - u: 130, v: 286
6 - u: 112, v: 312
7 - u: 38, v: 416
8 - u: 595, v: 290
9 - u: 579, v: 208
10 - u: 37, v: 275
11 - u: 531, v: 439
12 - u: 465, v: 361''')

PTS_GROUND_PLANE = np.array([
    [-1, 8],
    [1, 8],
    [-1, 16],
    [-4, 16],
    [4, 16],
    [-1, 7],
    [-1, 6],
    [-1, 4],
    [2, 5],
    [3, 8],
    [-2, 8],
    [1, 3],
    [1, 4],
]) * 304.8


print(PTS_IMAGE_PLANE)
assert (len(PTS_IMAGE_PLANE) == len(PTS_GROUND_PLANE))

image_path = "frame.jpg"
frame = cv2.imread(image_path)
cv2.imshow(IMAGE_WINDOW_NAME, frame)


def transform_uv_to_xy(h, u, v):
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


np_pts_ground = np.float32(PTS_GROUND_PLANE[:, np.newaxis, :])

np_pts_image = np.float32(PTS_IMAGE_PLANE[:, np.newaxis, :])

h, err = cv2.findHomography(np_pts_image, np_pts_ground)
print(h)
print(err)


# Mouse click event listener
def mouse_event_listener(event, u, v, flags, param):
    frame = param
    # For left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Transform to x y (world coordinate)
        x, y = transform_uv_to_xy(h, u, v)
        print(f"x: {x}, y: {y:}")  # Print world coordinate
        test_frame = cv2.circle(
            frame.copy(), (u, v), 10, (0, 0, 255), 2
        )  # Draw red circle

        # Print world coordinate on picture
        cv2.putText(
            test_frame,
            f"x: {x:.2f} y: {y:.2f}",
            (u - 120, v - 20),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            2,
        )
        cv2.imshow(IMAGE_WINDOW_NAME, test_frame)


# Set click callback
cv2.setMouseCallback(IMAGE_WINDOW_NAME, mouse_event_listener, frame)

# Wait for q or close to quit
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    try:
        cv2.getWindowProperty(IMAGE_WINDOW_NAME, 0)
    except cv2.error:
        break
cv2.destroyAllWindows()

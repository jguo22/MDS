import cv2
import numpy as np
import math


IMAGE_WINDOW_NAME = "manual calibration"


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


points_image = outputToArray('''0 - u: 295, v: 455
1 - u: 330, v: 455
2 - u: 364, v: 456
3 - u: 546, v: 115
4 - u: 702, v: 54
5 - u: 294, v: 110
6 - u: 462, v: 214
7 - u: 376, v: 160
8 - u: 260, v: 45
9 - u: 15, v: 433
10 - u: 12, v: 7
11 - u: 860, v: 11
12 - u: 845, v: 472
13 - u: 562, v: 424
''')

points_ground = np.array([
    [125, 18.9 * 3.5],
    [125, 18.9 * 2.5],
    [125, 18.9 * 1.5],
    [352, -18.9 * 4.5],
    [401.8, -198.35],
    [125 + 227, 18.9 * 4.5],
    [125 + 8 * 18.9, -18.9 * 1.5],
    [125 + 10 * 18.9, 18.9 * 1.5],
    [97 + 304.8, 97 + 18.9 / 2],
    [97 + 304.8, 97 + 175],
    [97 + 51, 97 - 304.8 - 115],
    [97 + 15, 97 - 304.8 - 40],
    [125 + 18.9, -18.9 * 4.5]
])

PTS_IMAGE_PLANE = outputToArray('''0 - u: 298, v: 420
1 - u: 319, v: 135
2 - u: 519, v: 139
3 - u: 530, v: 424
''')

PTS_GROUND_PLANE = np.array([
    [125 + 18.9, 18.9 * 3.5],
    [125 + 18.9 * 11, 18.9 * 3.5],
    [125 + 18.9 * 11, -18.9 * 3.5],
    [125 + 18.9, -18.9 * 3.5],
])

print(PTS_IMAGE_PLANE)
assert (len(PTS_IMAGE_PLANE) == len(PTS_GROUND_PLANE))

image_path = "bottom.jpg"
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

print(len(np_pts_ground))
h, err = cv2.findHomography(np_pts_image, np_pts_ground)
print("h and error")
print(h)
print(err)


# Mouse click event listener
def mouse_event_listener(event, u, v, flags, param):
    frame = param
    # For left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Transform to x y (world coordinate)
        x, y = transform_uv_to_xy(h, u, v)
        print("thing")
        print(f"x: {x}, y: {y:}")  # Print world coordinate
        print(f"u: {u}, v: {v}")
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


# for i in range(len(points_image)):
#     u, v = points_image[i]
#     predicted = transform_uv_to_xy(h, u, v)
#     marked = points_ground[i]
#     print(i)
#     print(distance(predicted, marked))

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


# Wait for q or close to quit
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    try:
        cv2.getWindowProperty(IMAGE_WINDOW_NAME, 0)
    except cv2.error:
        break
cv2.destroyAllWindows()

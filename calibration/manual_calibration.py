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


PTS_IMAGE_PLANE = outputToArray('''0 - u: 129, v: 280
1 - u: 164, v: 283
2 - u: 198, v: 285
3 - u: 232, v: 287
4 - u: 265, v: 290
5 - u: 298, v: 292
6 - u: 332, v: 295
7 - u: 364, v: 297
8 - u: 125, v: 286
9 - u: 119, v: 291
10 - u: 114, v: 299
11 - u: 106, v: 307
12 - u: 99, v: 315
13 - u: 92, v: 323
14 - u: 83, v: 333
15 - u: 75, v: 343
16 - u: 64, v: 355
17 - u: 52, v: 368
18 - u: 433, v: 396
19 - u: 318, v: 360
20 - u: 225, v: 331
21 - u: 391, v: 334''')

PTS_GROUND_PLANE = np.array([
    [0, 10],
    [1, 10],
    [2, 10],
    [3, 10],
    [4, 10],
    [5, 10],
    [6, 10],
    [7, 10],
    [0, 9],
    [0, 8],
    [0, 7],
    [0, 6],
    [0, 5],
    [0, 4],
    [0, 3],
    [0, 2],
    [0, 1],
    [0, 0],
    [7, 0],
    [5, 2],
    [3, 3],
    [7, 5],
]) * 18.5 + np.array([[-18.5 * 3.5, 279.4]])


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

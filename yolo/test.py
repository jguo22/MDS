import cv2
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO

# model = YOLO("best.pt")

# image = cv2.imread("frame.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# result = model.predict(image, verbose=False)[0]
# detections = sv.Detections.from_ultralytics(result)

# # Use Supervision annotator
# mask_annotator = sv.MaskAnnotator()
# box_annotator = sv.BoxAnnotator(thickness=2)
# annotated_image = box_annotator.annotate(
#     scene=image,
#     detections=detections
# )
# annotated_image = mask_annotator.annotate(
#     scene=annotated_image,
#     detections=detections
# )

# plt.figure(figsize=(10, 10))
# plt.imshow(annotated_image)
# plt.axis("off")
# plt.show()


import cv2
import numpy as np

def nothing(x):
    pass

# Open camera (change index if needed)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
# Set exposure time to 2^-7 = 1/128 second
cap.set(cv2.CAP_PROP_EXPOSURE, -7)
cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)  # Disable auto white balance
# Set white balance temperature to 4200K
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4200)

# Create windows
cv2.namedWindow("Original")
cv2.namedWindow("Mask")
cv2.namedWindow("HSV Controls")

# Create trackbars
cv2.createTrackbar("H Min", "HSV Controls", 38, 179, nothing)
cv2.createTrackbar("H Max", "HSV Controls", 146, 179, nothing)
cv2.createTrackbar("S Min", "HSV Controls", 0, 255, nothing)
cv2.createTrackbar("S Max", "HSV Controls", 57, 255, nothing)
cv2.createTrackbar("V Min", "HSV Controls", 111, 255, nothing)
cv2.createTrackbar("V Max", "HSV Controls", 227, 255, nothing)


IMAGE_WINDOW_NAME = "Original"

cap = cv2.VideoCapture(0)  # Change to your appropriate camera

# Mouse click event listener
def mouse_event_listener(event, u, v, flags, param):
    # For left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_event_listener.clicked_point = (u, v)


mouse_event_listener.clicked_point = None

# Set click callback
cv2.setMouseCallback(IMAGE_WINDOW_NAME, mouse_event_listener)

while True:

    image = cv2.imread("frame_20260123_130812_1769191692037.jpg")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H Min", "HSV Controls")
    h_max = cv2.getTrackbarPos("H Max", "HSV Controls")
    s_min = cv2.getTrackbarPos("S Min", "HSV Controls")
    s_max = cv2.getTrackbarPos("S Max", "HSV Controls")
    v_min = cv2.getTrackbarPos("V Min", "HSV Controls")
    v_max = cv2.getTrackbarPos("V Max", "HSV Controls")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    # forbidden_mask = cv2.dilate(mask, kernel_dilate)



    cv2.imshow("Original", image)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if mouse_event_listener.clicked_point is not None:
        u, v = mouse_event_listener.clicked_point
        # Draw red circle
        cv2.circle(
            image,
            (u, v),
            10,
            (0, 0, 255),
            2,
        )
        # Print HSV
        cv2.putText(
            image,
            f"hsv{hsv_frame[v][u]}",
            (u - 100, v + 80),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Original", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    try:
        cv2.getWindowProperty(IMAGE_WINDOW_NAME, 0)
    except cv2.error:
        break

cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

def detect_boundary_mask(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # These ranges WORK WELL for blue painter's tape / vinyl
    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask


def clean_boundary_mask(mask):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


def inflate_boundary(mask, pixels=30):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (pixels*2+1, pixels*2+1)
    )
    inflated = cv2.dilate(mask, kernel)
    return inflated

def debug_overlay(image, forbidden_mask):
    overlay = image.copy()
    overlay[forbidden_mask > 0] = (0, 0, 255)
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
if __name__ == "__main__":
    test_image = cv2.imread("frame.jpg")

    boundary_mask = detect_boundary_mask(test_image)
    boundary_mask = clean_boundary_mask(boundary_mask)
    forbidden_mask = inflate_boundary(boundary_mask, pixels=30)

    debug_image = debug_overlay(test_image, forbidden_mask)

    cv2.imshow("Boundary Mask", boundary_mask)
    cv2.imshow("Forbidden Mask", forbidden_mask)
    cv2.imshow("Debug Overlay", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

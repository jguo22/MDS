import cv2  # OpenCV version 2

IMAGE_WINDOW_NAME = "calibration"

image_path = "picked.jpg"
image = cv2.imread(image_path)
if image is None:
    raise
cv2.imshow(IMAGE_WINDOW_NAME, image)

click_count = 0


# Mouse click event listener
def mouse_event_listener(event, u, v, flags, param):
    global click_count
    image = param
    # For left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print clicked coordinate
        print(f"{click_count} - u: {u}, v: {v}")
        cv2.circle(image, (u, v), 10, (0, 0, 255), 2)  # Draw red circle
        # Print point order
        cv2.putText(
            image,
            str(click_count),
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            2,
        )
        cv2.imshow(IMAGE_WINDOW_NAME, image)
        # Increment click count
        click_count += 1


# Set click callback
cv2.setMouseCallback(IMAGE_WINDOW_NAME, mouse_event_listener, image)

# Wait for q or close to quit
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    try:
        cv2.getWindowProperty(IMAGE_WINDOW_NAME, 0)
    except cv2.error:
        break
cv2.destroyAllWindows()

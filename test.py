import numpy as np
import cv2 as cv


image_path = str("yolo/test.jpg")
print(f"Loading image: {image_path}")

frame = cv.imread(image_path)
print(frame)

# Black
color_range = np.max(frame[:, :, 1:], axis=2) - np.min(frame[:, :, 1:], axis=2)
range_thres = np.where(color_range < 10, 255, 0).astype(np.uint8)
hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
hsv_thres = cv.inRange(hsv_frame, (0, 0, 0), (180, 255, 50))

black_thres = cv.bitwise_and(hsv_thres, range_thres)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
black_thres = cv.dilate(black_thres, kernel, iterations=2)
black_thres = cv.erode(black_thres, kernel, iterations=1)

# Red
red_thres1 = cv.inRange(hsv_frame, (0, 100, 100), (10, 255, 255))
red_thres2 = cv.inRange(hsv_frame, (170, 100, 100), (180, 255, 255))

red_thres = cv.bitwise_or(red_thres1, red_thres2)
red_thres = cv.dilate(red_thres, kernel, iterations=2)
red_thres = cv.erode(red_thres, kernel, iterations=1)

# Combined
combined_thres = cv.bitwise_and(black_thres, red_thres)

cv.imshow("thres", combined_thres)
cv.imshow("image", frame)
cv.waitKey(0)

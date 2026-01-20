from yolo.segment import segmentImage
from getZones import getZones
import numpy as np
import cv2 as cv


image_path = str("yolo/test.jpg")
print(f"Loading image: {image_path}")

image = cv.imread(image_path)
if image is None:
    raise Exception("no image")

print("Running segmentation...")

result = segmentImage(image)

# Get quadrilaterals in xy coordinates, sorted by distance
quads_xy, class_names = getZones(result, image)


print("\nZones sorted by distance:")
for i, (quad, class_name) in enumerate(zip(quads_xy, class_names)):
    center_x = np.mean(quad[:, 0])
    center_y = np.mean(quad[:, 1])
    distance = np.sqrt(center_x**2 + center_y**2)
    print(f"{i}: {class_name:15s} - center=({center_x:7.1f}, {center_y:7.1f}) mm, distance={distance:7.1f} mm")

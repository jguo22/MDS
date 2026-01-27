import cv2 as cv
from pathlib import Path
from ultralytics.models.yolo import YOLO
from yolo.segment import segmentImage
from yolo.zone_utils import getZones

SCRIPT_DIR = Path(__file__).parent.absolute()
image_path = str(SCRIPT_DIR / "test.jpg")
print(f"Loading image: {image_path}")

image = cv.imread(image_path)
if image is None:
    raise Exception("no image")

print("Running segmentation...")
result = segmentImage(image)

# Display original segmentation
annotated_frame = result.plot(boxes=False)
cv.imshow("Original Segmentation", annotated_frame)

squares_xy, class_names = getZones(result, image)

print(squares_xy)
print(class_names)
cv.waitKey(0)

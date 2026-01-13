import cv2 as cv
from ultralytics import YOLO
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.absolute()


def segmentImage(image):
    results = model(image)
    annotated_frame = results[0].plot()
    cv.imshow("Segmentation", annotated_frame)
    cv.waitKey(10000)


if __name__ == "__main__":
    model = YOLO(str(SCRIPT_DIR / 'best.pt'))

    image_path = str(SCRIPT_DIR / "test.jpg")

    image = cv.imread(image_path)
    segmentImage(image)

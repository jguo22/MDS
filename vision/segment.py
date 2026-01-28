import cv2 as cv
from pathlib import Path
from ultralytics.models.yolo import YOLO


SCRIPT_DIR = Path(__file__).parent.absolute()

MODEL = YOLO(str(SCRIPT_DIR / 'best.pt'))
labels = MODEL.names
print(labels)


def wait_for_quit():
    """Waits for keypress. Returns True if 'q' pressed, False otherwise."""
    if cv.waitKey(0) & 0xFF == ord('q'):
        raise


def segmentImage(image):
    # model returns array of results
    # here, we only have one image so its an array of size 1
    result = MODEL(image, verbose=False)[0]
    return result


def getClassName(classidx: int):
    classname = labels[classidx]
    return classname


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Creates a colored overlay of the mask on the image.

    Args:
        image: BGR image
        mask: Binary mask (H, W) with values 0 or 255
        color: BGR color tuple (default: green)
        alpha: Transparency of overlay (0.0 to 1.0, default: 0.5)

    Returns:
        Blended image with mask overlay
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    result = cv.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result

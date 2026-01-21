import cv2 as cv
from pathlib import Path
import numpy as np
from ultralytics.models.yolo import YOLO

from mask_utils import calculateQuadFromMask, fixSegmentation, yoloMaskToBinary
from zone_utils import annotate_poly


SCRIPT_DIR = Path(__file__).parent.absolute()

MODEL = YOLO(str(SCRIPT_DIR / 'last.pt'))
print(MODEL.names)


def wait_for_quit():
    """Waits for keypress. Returns True if 'q' pressed, False otherwise."""
    if cv.waitKey(0) & 0xFF == ord('q'):
        raise


def compute_iou(mask1, mask2):
    """
    Computes Intersection over Union between two binary masks.

    Args:
        mask1, mask2: Binary masks (H, W) with values 0 or 255

    Returns:
        IoU score (0.0 to 1.0)
    """
    intersection = cv.bitwise_and(mask1, mask2)
    union = cv.bitwise_or(mask1, mask2)

    intersection_area = np.count_nonzero(intersection)
    union_area = np.count_nonzero(union)

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def click_hsv(event, x, y, flags, param):
    """Mouse callback to print HSV values at clicked point"""
    if event == cv.EVENT_LBUTTONDOWN:
        image, hsv = param
        h, s, v = hsv[y, x]
        bgr = image[y, x]
        print(f"Clicked ({x}, {y}): H={h}, S={s}, V={v} | BGR={bgr}")


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


def segmentImage(image):
    # model returns array of results
    # here, we only have one image so its an array of size 1
    result = MODEL(image, conf=0.25)[0]
    return result


def main():
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

    # Colors for different objects
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]

    # Create fresh images for this iteration
    all_quads_image = image.copy()
    all_quads_image2 = image.copy()
    all_quads_image3 = image.copy()

    # Process all masks with current parameters
    if result.masks is not None:
        for i, mask_orig in enumerate(result.masks):
            binary_mask = yoloMaskToBinary(mask_orig, image)

            # Apply fixSegmentation with current params
            tape_mask = fixSegmentation(image, binary_mask)

            quad = calculateQuadFromMask(binary_mask)
            quad2 = calculateQuadFromMask(tape_mask)

            if quad is None or quad2 is None:
                continue

            # Annotate with unique color
            color = colors[i % len(colors)]
            annotate_poly(all_quads_image, quad, color)
            annotate_poly(all_quads_image2, quad2, color)

            # Overlay segmentation
            all_quads_image3 = overlay_mask(
                all_quads_image3, tape_mask, color=color, alpha=0.4)

    # Display all windows
    cv.imshow("All Quadrilaterals", all_quads_image)
    cv.imshow("All Quadrilaterals 2", all_quads_image2)
    cv.imshow("All Segmentations", all_quads_image3)
    wait_for_quit()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

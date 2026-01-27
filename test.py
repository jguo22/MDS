import cv2
import numpy as np
from vision.segment import segmentImage
from vision.zone_utils import getZones


def detect_boundary_mask(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([44, 0, 101])
    upper_blue = np.array([146, 59, 227])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask


def clean_boundary_mask(mask):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def inflate_boundary(mask, pixels=30):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (pixels * 2 + 1, pixels * 2 + 1)
    )
    inflated = cv2.dilate(mask, kernel)
    return inflated


def debug_overlay(image, forbidden_mask):
    overlay = image.copy()
    overlay[forbidden_mask > 0] = (0, 0, 255)
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# TODO: THIS DOES NOT WORK
# Returns a list of (x, y) pixel coordinates representing the blue boundary


def get_boundaries(image):
    boundary_mask = detect_boundary_mask(image)
    boundary_mask = clean_boundary_mask(boundary_mask)
    ys, xs = np.where(boundary_mask > 0)

    pixels = list(zip(xs, ys))
    pixel_array = np.array(pixels)
    # return boundary_mask
    return pixel_array


if __name__ == "__main__":
    test_image = cv2.imread("test2.jpg")

    boundary_mask = detect_boundary_mask(test_image)
    boundary_mask = clean_boundary_mask(boundary_mask)
    # forbidden_mask = inflate_boundary(boundary_mask, pixels=30)

    debug_image = debug_overlay(test_image, boundary_mask)

    cv2.imshow("Boundary Mask", boundary_mask)
    # cv2.imshow("Forbidden Mask", forbidden_mask)
    cv2.imshow("Debug Overlay", debug_image)
    ys, xs = np.where(boundary_mask > 0)

    pixels = list(zip(xs, ys))
    pixel_array = np.array(pixels)

    print(f"Detected {len(pixels)} boundary pixels.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

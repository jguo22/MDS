from mask_utils import getSmoothRegionFromMask
import numpy as np


def getBottomCenterPixel(mask):
    """
    Gets the bottom center pixel from a segmentation mask.

    Args:
        mask: Binary mask (H, W) with values 0 or 255

    Returns:
        tuple: (x, y) coordinates of bottom center pixel, or None if mask is empty
    """
    # Get smooth region from mask
    roi_mask = getSmoothRegionFromMask(mask)

    # Find all non-zero pixels
    non_zero = np.argwhere(roi_mask > 0)

    if len(non_zero) == 0:
        return None

    # non_zero is array of [y, x] coordinates
    # Find maximum y value (bottom-most row)
    max_y = np.max(non_zero[:, 0])

    # Get all pixels in the bottom row
    bottom_pixels = non_zero[non_zero[:, 0] == max_y]

    # Find center x coordinate among bottom pixels
    x_coords = bottom_pixels[:, 1]
    center_x = int(np.mean(x_coords))

    return (center_x, max_y)

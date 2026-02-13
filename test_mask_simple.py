#!/usr/bin/env python3
"""Minimal test to check YOLO mask extraction."""

import cv2
import numpy as np
from ultralytics import YOLO
import sys


def simplify_polygon(polygon, epsilon_factor=0.02):
    """
    Simplify a polygon using Douglas-Peucker algorithm.

    Args:
        polygon: Numpy array of shape (N, 2) with polygon vertices
        epsilon_factor: Approximation accuracy factor (default: 0.02)
                       epsilon = epsilon_factor * perimeter

    Returns:
        Simplified polygon as numpy array of shape (M, 2) where M <= N
    """
    # Calculate perimeter
    perimeter = cv2.arcLength(polygon.astype(np.float32), closed=True)

    # Calculate epsilon
    epsilon = epsilon_factor * perimeter

    # Approximate polygon
    approx = cv2.approxPolyDP(polygon.astype(np.float32), epsilon, closed=True)

    # Reshape from (N, 1, 2) to (N, 2)
    return approx.reshape(-1, 2)

if len(sys.argv) < 2:
    print("Usage: python3 test_mask_simple.py <image_path>")
    sys.exit(1)

# Load image
image = cv2.imread(sys.argv[1])
if image is None:
    print(f"Error: Cannot load {sys.argv[1]}")
    sys.exit(1)

print(f"Image shape: {image.shape}")

# Load YOLO model and run inference
model = YOLO('vision/best.pt')
results = model(image, verbose=False)
result = results[0]

print(f"\nMasks detected: {len(result.masks) if result.masks else 0}")

if result.masks is None:
    print("No masks found!")
    sys.exit(0)

# Create visualization
vis = image.copy()

for i, mask in enumerate(result.masks):
    # Get mask data
    mask_data = mask.data[0].cpu().numpy()
    print(f"\nMask {i}:")
    print(f"  mask.data shape: {mask_data.shape}")

    # Convert to uint8
    mask_uint8 = (mask_data * 255).astype(np.uint8)

    # Method 1: Resize with INTER_LINEAR (default)
    mask_linear = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]))

    # Method 2: Resize with INTER_NEAREST
    mask_nearest = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # Use mask.xy if available (YOLO provides polygon directly - most accurate)
    if hasattr(mask, 'xy') and len(mask.xy) > 0:
        poly = mask.xy[0]  # Get first polygon
        pts = poly.cpu().numpy() if hasattr(poly, 'cpu') else poly

        print(f"  mask.xy polygon: {len(pts)} points")

        # Simplify polygon using the function
        simplified = simplify_polygon(pts, epsilon_factor=0.02)

        # Draw simplified polygon in green
        cv2.polylines(vis, [simplified.astype(np.int32)], True, (0, 255, 0), 3)
        print(f"  Original: {len(pts)} points")
        print(f"  Simplified (eps=0.02): {len(simplified)} vertices")

# Show results
cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
cv2.imshow("Test", vis)
print("\n=== Legend ===")
print("Green = Simplified polygon (epsilon=0.02)")
print("Blue = Simplified with epsilon=0.03 (simpler)")
print("\nPress any key to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()

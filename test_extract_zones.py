#!/usr/bin/env python3
"""Extract zones from YOLO results using mask.xy polygons."""

import cv2
import numpy as np
from ultralytics import YOLO
from vision.pixelTo3D import transform_uv_to_xy
from colors import ZONE_CLASS_NAMES
import sys


def simplify_polygon(polygon, epsilon_factor=0.02):
    """Simplify polygon using Douglas-Peucker algorithm."""
    perimeter = cv2.arcLength(polygon.astype(np.float32), closed=True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(polygon.astype(np.float32), epsilon, closed=True)
    return approx.reshape(-1, 2)


def extract_zones(result, image, is_top=True, epsilon_factor=0.02):
    """
    Extract zones from YOLO result using mask.xy polygons and convert to world coordinates.

    Args:
        result: YOLO result object with masks
        image: Original image (for reference, not used for coordinates)
        is_top: True for top camera, False for bottom camera
        epsilon_factor: Polygon simplification factor (default: 0.02)

    Returns:
        List of dicts with keys:
        - 'polygon_xy': Zone polygon in world coordinates (mm), shape (N, 2)
        - 'polygon_uv': Zone polygon in pixel coordinates, shape (N, 2)
        - 'class_name': Zone class name (e.g., 'Green Zone')
        - 'confidence': Detection confidence
    """
    zones = []

    if result.masks is None or len(result.masks) == 0:
        return zones

    for i, mask in enumerate(result.masks):
        # Get class info
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]
        confidence = float(result.boxes.conf[i])

        # Skip if not a zone
        if class_name not in ZONE_CLASS_NAMES:
            continue

        # Get polygon from mask.xy
        if not hasattr(mask, 'xy') or len(mask.xy) == 0:
            continue

        # Extract polygon in pixel coordinates
        poly_uv = mask.xy[0]
        if hasattr(poly_uv, 'cpu'):
            poly_uv = poly_uv.cpu().numpy()
        else:
            poly_uv = np.array(poly_uv)

        if len(poly_uv) < 3:
            continue

        # Simplify polygon
        poly_uv_simplified = simplify_polygon(poly_uv, epsilon_factor)

        # Transform to world coordinates
        poly_xy = []
        for point in poly_uv_simplified:
            u, v = point[0], point[1]
            xy = transform_uv_to_xy(u, v, is_top)
            if xy is not None:
                poly_xy.append(xy)

        if len(poly_xy) < 3:
            continue

        poly_xy = np.array(poly_xy)

        zones.append({
            'polygon_xy': poly_xy,
            'polygon_uv': poly_uv_simplified,
            'class_name': class_name,
            'confidence': confidence,
            'num_vertices': len(poly_xy)
        })

        print(f"{class_name}: {len(poly_xy)} vertices (confidence: {confidence:.2f})")

    return zones


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_extract_zones.py <image_path>")
        sys.exit(1)

    # Load image
    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"Error: Cannot load {sys.argv[1]}")
        sys.exit(1)

    print(f"Image shape: {image.shape}\n")

    # Run YOLO
    model = YOLO('vision/best.pt')
    results = model(image, verbose=False)
    result = results[0]

    # Extract zones
    print("Extracted zones:")
    zones = extract_zones(result, image, is_top=True, epsilon_factor=0.02)

    print(f"\nTotal zones: {len(zones)}\n")

    # Visualize
    vis = image.copy()
    colors = {
        'Green Zone': (0, 255, 0),
        'Red Zone': (0, 0, 255),
        'Golden Zone': (0, 215, 255),
    }

    for zone in zones:
        color = colors.get(zone['class_name'], (255, 255, 255))
        poly = zone['polygon_uv'].astype(np.int32)
        cv2.polylines(vis, [poly], True, color, 3)

        # Label
        center = np.mean(poly, axis=0).astype(np.int32)
        cv2.putText(vis, f"{zone['class_name']}\n{zone['num_vertices']}v",
                   tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.namedWindow("Zones", cv2.WINDOW_NORMAL)
    cv2.imshow("Zones", vis)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

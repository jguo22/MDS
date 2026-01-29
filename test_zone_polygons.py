#!/usr/bin/env python3
"""
Test script for zone polygon approximation.

Tests the getZonesPolygon function on images and visualizes the results.
"""

import cv2
import numpy as np
import argparse
from vision.segment import segmentImage
from vision.zone_utils import getZonesPolygon, visualize_convex_hulls
from colors import ZONE_CLASS_NAMES


def visualize_polygons(image, polygons, class_names, color=(0, 255, 0)):
    """
    Draw polygons on the image.

    Args:
        image: Image to draw on (BGR format)
        polygons: List of polygon arrays (each is (N, 2))
        class_names: List of class names for each polygon
        color: BGR color tuple for drawing

    Returns:
        Image with polygons drawn
    """
    output = image.copy()

    zone_colors = {
        "Green Zone": (0, 255, 0),      # Green
        "Red Zone": (0, 0, 255),        # Red
        "Golden Zone": (0, 215, 255),   # Gold
    }

    for polygon, name in zip(polygons, class_names):
        # Get color for this zone type
        poly_color = zone_colors.get(name, color)

        # Draw polygon outline
        pts = polygon.astype(np.int32)
        cv2.polylines(output, [pts], isClosed=True, color=poly_color, thickness=3)

        # Draw vertices
        for i, point in enumerate(pts):
            cv2.circle(output, tuple(point), 5, poly_color, -1)
            # Label vertices
            cv2.putText(output, str(i), (point[0] + 8, point[1] + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, poly_color, 1)

        # Draw zone name at center
        center = np.mean(pts, axis=0).astype(np.int32)
        cv2.putText(output, name, tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, poly_color, 2)

    return output


def test_image(image_path, epsilon_factor=0.02, show_hulls=False):
    """
    Test polygon approximation on a single image.

    Args:
        image_path: Path to test image
        epsilon_factor: Polygon simplification factor (lower = more vertices)
        show_hulls: If True, show convex hulls overlay
    """
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"Epsilon factor: {epsilon_factor}")
    print(f"{'='*60}\n")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Run segmentation
    print("Running YOLO segmentation...")
    result = segmentImage(image)

    if result.masks is None:
        print("No masks detected!")
        return

    print(f"Found {len(result.masks)} masks")

    # Get polygons in PIXEL coordinates for visualization
    print(f"\nExtracting polygons with epsilon_factor={epsilon_factor}...")
    polygons_pixel, class_names, confidences = getZonesPolygon(
        result, image, is_top=True, epsilon_factor=epsilon_factor, return_pixel_coords=True)

    print(f"\nDetected {len(polygons_pixel)} zones:")
    for i, (name, conf, polygon) in enumerate(zip(class_names, confidences, polygons_pixel)):
        print(f"  {i+1}. {name} - {len(polygon)} vertices, confidence: {conf:.2f}")

    # Create visualizations
    # 1. YOLO segmentation with masks
    vis_yolo = result.plot(boxes=True, masks=True, conf=True, line_width=2, labels=True)

    # 2. Convex hulls overlay
    vis_hulls = visualize_convex_hulls(image, result, color=(255, 255, 0), thickness=2)

    # 3. Draw polygons (already in pixel coordinates from getZonesPolygon)
    vis_polygons = visualize_polygons(image, polygons_pixel, class_names)

    # 4. Comparison view: Convex hulls + Polygons overlaid
    vis_comparison = image.copy()
    # Draw convex hulls in cyan
    vis_comparison = visualize_convex_hulls(vis_comparison, result, color=(255, 255, 0), thickness=2)
    # Draw polygons on top in their zone colors
    vis_comparison = visualize_polygons(vis_comparison, polygons_pixel, class_names)

    # Display results
    # Combine in 2x2 grid
    h, w = image.shape[:2]
    combined = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    combined[0:h, 0:w] = cv2.resize(vis_yolo, (w, h))
    combined[0:h, w:2*w] = cv2.resize(vis_hulls, (w, h))
    combined[h:2*h, 0:w] = cv2.resize(vis_polygons, (w, h))
    combined[h:2*h, w:2*w] = cv2.resize(vis_comparison, (w, h))

    # Add labels
    cv2.putText(combined, "YOLO Segmentation", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Convex Hulls (Cyan)", (w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, f"Polygons (eps={epsilon_factor})", (10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Hull + Polygon Overlay", (w + 10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show
    window_name = "Zone Polygon Approximation Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined)

    print("\nPress any key to continue, 'q' to quit, 's' to save...")
    key = cv2.waitKey(0)

    if key == ord('s'):
        output_path = image_path.replace('.', f'_polygons_eps{epsilon_factor}.')
        cv2.imwrite(output_path, combined)
        print(f"Saved to: {output_path}")

    cv2.destroyAllWindows()

    return key == ord('q')


def main():
    parser = argparse.ArgumentParser(description="Test zone polygon approximation")
    parser.add_argument("images", nargs='+', help="Image file(s) to test")
    parser.add_argument("--epsilon", type=float, default=0.02,
                       help="Epsilon factor for polygon approximation (default: 0.02)")
    parser.add_argument("--show-hulls", action="store_true",
                       help="Show convex hulls overlay")
    args = parser.parse_args()

    print(f"\nTesting {len(args.images)} image(s)")
    print(f"Settings: epsilon_factor={args.epsilon}, show_hulls={args.show_hulls}")

    for image_path in args.images:
        should_quit = test_image(image_path, args.epsilon, args.show_hulls)
        if should_quit:
            break

    print("\nTest complete!")


if __name__ == "__main__":
    main()

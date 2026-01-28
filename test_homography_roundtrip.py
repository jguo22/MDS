"""
Test round-trip transformation: pixel → world → pixel

Verifies that transform_uv_to_xy and world_to_pixel are inverse operations.
"""

import numpy as np
from vision.pixelTo3D import transform_uv_to_xy, H_TOP, H_BOTTOM
from vision.relativeCoordinates import world_to_pixel


def test_roundtrip(u, v, h_matrix, camera_name):
    """Test pixel → world → pixel round-trip transformation."""
    print(f"\n{camera_name} Camera:")
    print(f"  Original pixel: ({u}, {v})")

    # Forward: pixel → world
    is_top = (h_matrix is H_TOP)
    xy = transform_uv_to_xy(u, v, is_top)

    if xy is None:
        print(f"  ❌ Forward transform failed (point behind camera)")
        return False

    x, y = xy
    print(f"  World coords: ({x:.2f}, {y:.2f}) mm")

    # Backward: world → pixel
    uv_result = world_to_pixel((x, y), h_matrix)

    if uv_result is None:
        print(f"  ❌ Backward transform failed")
        return False

    u_result, v_result = uv_result
    print(f"  Result pixel: ({u_result}, {v_result})")

    # Calculate error
    error_u = abs(u - u_result)
    error_v = abs(v - v_result)
    error_total = np.sqrt(error_u**2 + error_v**2)

    print(f"  Error: ({error_u}, {error_v}) pixels, total: {error_total:.2f} pixels")

    # Check if error is acceptable (< 2 pixels is good)
    if error_total < 2.0:
        print(f"  ✅ Round-trip successful!")
        return True
    elif error_total < 5.0:
        print(f"  ⚠️  Round-trip has moderate error")
        return True
    else:
        print(f"  ❌ Round-trip has large error!")
        return False


def main():
    print("=" * 60)
    print("Testing Homography Round-Trip Transformation")
    print("=" * 60)

    # Test points at various locations in the image
    test_points = [
        (320, 240, "center"),           # Image center
        (432, 240, "center-right"),     # Center right
        (200, 240, "center-left"),      # Center left
        (432, 360, "bottom-center"),    # Bottom center
        (432, 120, "top-center"),       # Top center
        (100, 100, "top-left"),         # Top left corner
        (764, 380, "bottom-right"),     # Bottom right corner
    ]

    results_top = []
    results_bottom = []

    # Test top camera
    print("\n" + "=" * 60)
    print("TOP CAMERA TESTS")
    print("=" * 60)
    for u, v, label in test_points:
        print(f"\n--- Testing {label} ---")
        success = test_roundtrip(u, v, H_TOP, "Top")
        results_top.append((label, success))

    # Test bottom camera
    print("\n" + "=" * 60)
    print("BOTTOM CAMERA TESTS")
    print("=" * 60)
    for u, v, label in test_points:
        print(f"\n--- Testing {label} ---")
        success = test_roundtrip(u, v, H_BOTTOM, "Bottom")
        results_bottom.append((label, success))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nTop Camera:")
    for label, success in results_top:
        status = "✅" if success else "❌"
        print(f"  {status} {label}")

    print("\nBottom Camera:")
    for label, success in results_bottom:
        status = "✅" if success else "❌"
        print(f"  {status} {label}")

    # Overall result
    all_success = all(s for _, s in results_top) and all(s for _, s in results_bottom)

    print("\n" + "=" * 60)
    if all_success:
        print("✅ All tests passed! Homography round-trip is working correctly.")
    else:
        print("⚠️  Some tests failed. Check homography matrices.")
    print("=" * 60)


if __name__ == "__main__":
    main()

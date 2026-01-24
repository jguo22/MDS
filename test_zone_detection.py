"""
Example usage of isPointInQuad for zone detection.
"""
import numpy as np
from yolo.zone_utils import isPointInPoly, getQuadCenter

# Example zone quadrilateral (in mm, from world coordinates)
# Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
zone_quad = np.array([
    [1000.0, 500.0],   # Bottom-left
    [1500.0, 500.0],   # Bottom-right
    [1500.0, 1000.0],  # Top-right
    [1000.0, 1000.0]   # Top-left
])

# Test points
robot_position_inside = (1250.0, 750.0)  # Center of zone
robot_position_outside = (2000.0, 750.0)  # Outside zone
robot_position_on_edge = (1000.0, 750.0)  # On boundary

# Check if points are in zone
print(f"Point {robot_position_inside} in zone: {isPointInPoly(robot_position_inside, zone_quad)}")
print(f"Point {robot_position_outside} in zone: {isPointInPoly(robot_position_outside, zone_quad)}")
print(f"Point {robot_position_on_edge} in zone: {isPointInPoly(robot_position_on_edge, zone_quad)}")

# Get zone center
center = getQuadCenter(zone_quad)
print(f"\nZone center: {center}")
print(f"Center in zone: {isPointInPoly(center, zone_quad)}")

from typing import Tuple
import math
from config import WHEEL_D, TICK_ROTATION, BASE_RATIO, BASE_D


def get_forward_mm(distance_mm: float) -> Tuple[float, float, float]:
    distance = distance_mm / (WHEEL_D * math.pi) * TICK_ROTATION
    return (1, 1, distance)


def get_rotate(theta: float) -> Tuple[float, float, float]:
    # make into range -pi to pi
    theta = theta % (2 * math.pi)
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < -math.pi:
        theta += 2 * math.pi

    # CCW is positive angle
    if theta >= 0:
        return (-1, 1, TICK_ROTATION / BASE_RATIO * theta / (2 * math.pi))
    else:
        return (1, -1, TICK_ROTATION / BASE_RATIO * -theta / (2 * math.pi))


def get_arc(dx: float, dy: float) -> Tuple[float, float, float]:
    """
    Calculate arc motion to reach relative point (x, y) in one smooth arc.

    Args:
        x: forward distance in mm (positive = forward, negative = backward)
        y: lateral distance in mm (positive = left, negative = right)

    Returns:
        (left_coef, right_coef, distance_ticks)
        - left_coef: left wheel speed coefficient
        - right_coef: right wheel speed coefficient
        - distance_ticks: arc distance in encoder ticks

    Example:
        # Move to point 500mm forward, 200mm to the left
        left_c, right_c, dist = get_arc(500, 200)
    """
    # Handle straight-line case (y ≈ 0)
    if abs(dy) < 1e-3:
        return get_forward_mm(dx)

    # Calculate radius of circular arc
    # Center of circle is at (0, R) for y > 0 (left turn)
    # or (0, R) for y < 0 (right turn, R will be negative)
    R = (dx * dx + dy * dy) / (2 * dy)

    # Calculate arc angle using chord length
    dist_straight = math.sqrt(dx * dx + dy * dy)
    theta = 2 * math.asin(dy / dist_straight)  # Signed angle

    # Arc length at robot center (always positive)
    arc_length_mm = abs(R * theta)

    # Differential drive wheel coefficients
    # For R > 0 (left turn): left wheel slower, right wheel faster
    # For R < 0 (right turn): left wheel faster, right wheel slower
    left_coef = 1.0 - BASE_D / (2.0 * R)
    right_coef = 1.0 + BASE_D / (2.0 * R)

    # Convert arc length to encoder ticks
    distance_ticks = arc_length_mm / (WHEEL_D * math.pi) * TICK_ROTATION

    return (left_coef, right_coef, distance_ticks)

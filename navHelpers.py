from typing import Tuple
import math
from config import WHEEL_D, TICK_ROTATION, BASE_RATIO


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

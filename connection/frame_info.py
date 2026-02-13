"""
Frame information data structure for robot state and sensor data.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FrameInfo:
    """Represents frame information with robot state and sensor data."""
    frame_top: np.ndarray
    frame_bottom: np.ndarray
    frame_id: int
    x: float
    y: float
    theta: float
    gripperHeight: float
    gripperAngle: float
    scooperAngle: float
    distanceSensed: float
    isMoving: bool
    lastCompletedCommandId: int

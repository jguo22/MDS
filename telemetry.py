from streamer import Streamer
import time
import cv2
import random
import math
from typing import Any, Dict, Optional


class Telemetry:
    """Wrapper around :class:`streamer.Streamer` for MASLAB telemetry."""

    def __init__(self, stream: Optional[Streamer] = None) -> None:
        self.stream = stream or Streamer()
        self._last_data: Dict[str, Any] = {}

    def set_img(self, img: cv2.Mat) -> None:
        """Forward the most recent camera frame to the dashboard."""
        self.stream.set_img(img)

    def set_data(self, data: Dict[str, Any]) -> None:
        """Update arbitrary JSON-serialisable telemetry data."""
        self._last_data = data
        self.stream.set_data(data)

    def set_odometry(self, x: float, y: float, theta: float) -> None:
        """Update robot pose and geometry overlays."""
        self.stream.set_odometry({
            "x": x,
            "y": y,
            "theta": theta
        })

    def set_dict(self, stuff) -> None:
        self.stream.set_odometry(stuff)


if __name__ == "__main__":
    stream = Telemetry()

    pos = [0, 0, random.uniform(0, 360)]  # x, y, theta

    # display an image
    stream.set_img(cv2.imread("frame.jpg"))

    # continuously update data and odometry
    curr_data: Dict[str, Any] = {}
    while True:
        pos[0] += math.cos(math.radians(pos[2])) * 0.2
        pos[1] += math.sin(math.radians(pos[2])) * 0.2
        pos[2] += random.uniform(-5, 5)
        odom = {
            "x": pos[0],
            "y": pos[1],
            "theta": pos[2],
            "circles": [
                {"x": 40, "y": 20, "c": "red"},
                {"x": 130, "y": -40, "c": "green"},
                {"x": 40, "y": 100, "c": "blue"},
            ],
            "lines": [
                {"x1": 0, "y1": 0, "x2": 100, "y2": 0, "c": "red"},
            ],
        }
        curr_data["timestamp"] = time.time()
        curr_data["random_value"] = random.random()
        curr_data["odometry"] = odom

        # stream data and odometry
        stream.set_data(curr_data)
        stream.set_dict(odom)

        time.sleep(0.05)

import cv2
import numpy as np
from nav import Nav
from typing import Optional, Tuple
from MovementCommander import MovementCommander
from FrameProcessor import FrameProcessor


class ClickProcessor(FrameProcessor):
    def __init__(
            self,
            movementCommander: MovementCommander,
            window_name: str = "Pi Camera"):
        self.movementCommander = movementCommander
        self.window_name = window_name
        self.frame_size = (1000, 1000)  # (width, height)
        # list of time of starting path, l_c, r_c, dist

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # using this for calcuations only
        self.nav = Nav()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert to normalized coordinates (0-1)
            x_norm = x / (self.frame_size[1] - 1)
            y_norm = y / (self.frame_size[0] - 1)
            # Scale to range of [-scale, scale] (centered at 0)
            scale = 10
            x_scaled = (x_norm * scale * 2) - scale
            y_scaled = -((y_norm * scale * 2) - scale)
            print(
                f"Click: ({x}, {y}) -> Normalized: ({x_norm:.3f}, {y_norm:.3f}) -> Scaled: ({x_scaled:.1f}, {y_scaled:.1f})")

            self.movementCommander.queue_xy(x_scaled, y_scaled)

    def process(self, frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float, float]]:
        # Update frame dimensions
        self.frame_size = (frame.shape[1], frame.shape[0])
        return None

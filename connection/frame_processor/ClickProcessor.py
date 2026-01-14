import cv2
import numpy as np
from typing import Optional, Tuple
from connection.ComputerReceiver import ComputerReceiver
from .FrameProcessor import FrameProcessor
from pixelTo3D import pixel_to_robot_horizontal


class ClickProcessor(FrameProcessor):
    def __init__(
            self,
            computerReceiver: ComputerReceiver,
            window_name: str = "Pi Camera"):
        self.computerReceiver = computerReceiver
        self.window_name = window_name
        self.frame_size = (640, 480)  # (width, height)
        # list of time of starting path, l_c, r_c, dist

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert to normalized coordinates (0-1)
            x_norm = (x / (self.frame_size[1])) + 1 / self.frame_size[1] / 2
            y_norm = y / (self.frame_size[0]) + 1 / self.frame_size[0] / 2
            x = x_norm * 640
            y = y_norm * 480
            print(
                f"Click: ({x}, {y}) -> Normalized: ({x_norm:.3f}")

            x_scaled, y_scaled = pixel_to_robot_horizontal(x, y)

            print(f'({x_scaled}, {y_scaled})')

            self.computerReceiver.send_xy(x_scaled, y_scaled)

    def process(self, frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float, float]]:
        # Update frame dimensions
        self.frame_size = (frame.shape[1], frame.shape[0])
        return None

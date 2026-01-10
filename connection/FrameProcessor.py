from typing import Optional, Tuple
import cv2
import numpy as np


class FrameProcessor:
    """
    Modular frame processor for detecting objects and computing coordinates.

    Subclass this to implement custom processing logic.
    """

    def process(self, frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float]]:
        """
        Process a frame and return coordinates.

        Args:
                frame: BGR image as numpy array
                frame_id: Frame sequence number

        Returns:
                (x, y) coordinates or None
        """
        raise NotImplementedError


class ClickProcessor(FrameProcessor):
    """Simple processor that returns mouse click coordinates."""

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_coords = (float(x), float(y))
            print(f"Click: ({x}, {y})")

            # Convert to normalized coordinates (0-1)
            # -1 to ensure 1.0 is at the last pixel
            x_norm = x / (self.width - 1)
            y_norm = y / (self.height - 1)
            self.click_coords = (x_norm, y_norm)
            print(
                f"Click: ({x}, {y}) -> Normalized: ({x_norm:.3f}, {y_norm:.3f})")

    def process(self, frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float]]:
        # Return and clear click coordinates
        coords = self.click_coords
        self.click_coords = None
        return coords

    def __init__(self, window_name: str = "Pi Camera"):
        self.window_name = window_name
        self.click_coords: Optional[Tuple[float, float]] = None
        self._setup = False
        self.width = 1000
        self.height = 1000
        cv2.setMouseCallback(self.window_name, self._mouse_callback())


class CenterProcessor(FrameProcessor):
    """Processor that always returns the frame center."""

    def process(self, frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float]]:
        h, w = frame.shape[:2]
        return (w / 2.0, h / 2.0)

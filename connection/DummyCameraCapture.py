from typing import Optional
import numpy as np
import cv2
import time


class DummyCameraCapture:
    """Dummy camera capture that returns generated frames for testing."""

    def __init__(self, source: str, width: int = 864, height: int = 480, threaded: bool = True):
        """
        Initialize dummy camera capture.

        Args:
            source: Camera source identifier (ignored, for compatibility)
            width: Frame width
            height: Frame height
            threaded: Threading parameter (ignored, for compatibility)
        """
        self.source = source
        self.width = width
        self.height = height
        self.threaded = threaded
        self._is_open = False
        self._frame_count = 0

    def open(self) -> bool:
        """Open the dummy camera. Returns True."""
        self._is_open = True
        self._frame_count = 0
        print(f"Dummy camera opened: {self.source}")
        print(f"Actual FPS: 30")
        print(f"Actual Width: {self.width}")
        print(f"Actual Height: {self.height}")
        print(f"Actual Buffer: 1")
        return True

    def read(self) -> Optional[np.ndarray]:
        """
        Generate and return a test frame.

        Returns a BGR numpy array with visual patterns for testing.
        """
        if not self._is_open:
            return None

        # Create a frame with a gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Add gradient background (blue to green)
        for y in range(self.height):
            intensity = int((y / self.height) * 255)
            frame[y, :] = [intensity, 255 - intensity, 128]

        # Add some geometric patterns for visual reference
        # Draw diagonal lines
        cv2.line(frame, (0, 0), (self.width, self.height), (255, 255, 255), 2)
        cv2.line(frame, (self.width, 0), (0, self.height), (255, 255, 255), 2)

        # Draw center crosshair
        center_x, center_y = self.width // 2, self.height // 2
        cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), (0, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), (0, 255, 255), 2)

        # Add corner markers
        marker_size = 30
        cv2.rectangle(frame, (0, 0), (marker_size, marker_size), (0, 0, 255), -1)
        cv2.rectangle(frame, (self.width - marker_size, 0), (self.width, marker_size), (0, 255, 0), -1)
        cv2.rectangle(frame, (0, self.height - marker_size), (marker_size, self.height), (255, 0, 0), -1)
        cv2.rectangle(frame, (self.width - marker_size, self.height - marker_size),
                     (self.width, self.height), (255, 255, 0), -1)

        # Add frame counter and timestamp
        self._frame_count += 1
        timestamp = time.time()
        text = f"Frame: {self._frame_count} | {self.source}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        time_text = f"Time: {timestamp:.2f}"
        cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add resolution info
        res_text = f"{self.width}x{self.height}"
        cv2.putText(frame, res_text, (self.width - 150, self.height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def is_open(self) -> bool:
        """Check if dummy camera is currently open."""
        return self._is_open

    def reopen(self) -> bool:
        """Close and reopen the dummy camera. Returns True."""
        self.close()
        return self.open()

    def close(self):
        """Close the dummy camera."""
        self._is_open = False
        print(f"Dummy camera closed: {self.source}")

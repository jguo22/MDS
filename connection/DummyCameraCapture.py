from typing import Optional
import numpy as np


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
        print("Actual FPS: 30")
        print(f"Actual Width: {self.width}")
        print(f"Actual Height: {self.height}")
        print("Actual Buffer: 1")
        return True

    def read(self) -> Optional[np.ndarray]:
        """
        Generate and return a test frame.

        Returns a BGR numpy array (all black).
        """
        if not self._is_open:
            return None

        # Create an all black frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self._frame_count += 1

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

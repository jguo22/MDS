from typing import Optional
import cv2
import numpy as np

from . import config


class CameraCapture:
    """Camera capture for USB cameras."""

    def __init__(self, source: str = "usb0",
                 width: int = config.FRAME_WIDTH,
                 height: int = config.FRAME_HEIGHT):
        """
        Initialize camera capture.

        Args:
            source: Camera source - "usb0", "usb1", etc. or device index
            width: Frame width
            height: Frame height
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None

    def open(self) -> bool:
        """Open the camera. Returns True if successful."""
        return self._open_usb()

    def _open_usb(self) -> bool:
        """Open USB camera."""
        if self.source.startswith("usb"):
            index = int(self.source[3:])
        else:
            index = int(self.source) if self.source.isdigit() else 0

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # disable auto exposure and white balance to prevent messing up calibration
        # TODO: maybe remove this later
        # Set to manual exposure mode with 0.25 "magic number"
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # Set exposure time to 2^-7 = 1/128 second
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)  # Disable auto white balance
        # Set white balance temperature to 4200K
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4200)
        return True

    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the camera. Returns BGR numpy array or None."""
        try:
            if self.cap is not None:
                ret, frame = self.cap.read()
                return frame if ret else None
        except Exception as e:
            print(f"Camera read error: {e}")
            return None
        return None

    def is_open(self) -> bool:
        """Check if camera is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def reopen(self) -> bool:
        """Close and reopen the camera. Returns True if successful."""
        self.close()
        return self.open()

    def close(self):
        """Release camera resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

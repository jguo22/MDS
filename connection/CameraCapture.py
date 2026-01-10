from typing import Optional
import cv2
import numpy as np

from . import config


class CameraCapture:
    """Modular camera capture supporting USB and PiCamera."""

    def __init__(self, source: str = "usb0",
                 width: int = config.FRAME_WIDTH,
                 height: int = config.FRAME_HEIGHT):
        """
        Initialize camera capture.

        Args:
            source: Camera source - "usb0", "usb1", "picamera0", etc.
            width: Frame width
            height: Frame height
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.picam = None

    def open(self) -> bool:
        """Open the camera. Returns True if successful."""
        if self.source.startswith("picamera"):
            return self._open_picamera()
        elif self.source.startswith("usb"):
            return self._open_usb()
        else:
            # Assume it's a device index or path
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
        return True

    def _open_picamera(self) -> bool:
        """Open PiCamera using picamera2."""
        try:
            from picamera2 import Picamera2
            index = int(self.source[9:]) if len(self.source) > 9 else 0
            self.picam = Picamera2(index)
            camera_config = self.picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam.configure(camera_config)
            self.picam.start()
            return True
        except ImportError:
            print("picamera2 not installed. Install with: pip install picamera2")
            return False
        except Exception as e:
            print(f"Failed to open PiCamera: {e}")
            return False

    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the camera. Returns BGR numpy array or None."""
        try:
            if self.picam is not None:
                frame = self.picam.capture_array()
                # Convert RGB to BGR for OpenCV compatibility
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif self.cap is not None:
                ret, frame = self.cap.read()
                return frame if ret else None
        except Exception as e:
            print(f"Camera read error: {e}")
            return None
        return None

    def close(self):
        """Release camera resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.picam is not None:
            try:
                self.picam.stop()
            except Exception:
                pass
            self.picam = None

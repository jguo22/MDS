from typing import Optional
import cv2
import numpy as np
import config


class CameraCapture:
    """Camera capture for USB cameras."""

    def __init__(self, source: str,
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
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # disable auto exposure and white balance to prevent messing up calibration
        # Set to manual exposure mode with 0.25 "magic number"
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        # Set exposure time to 2^-7 = 1/128 second
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        # Disable auto white balance
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        # Set white balance temperature to 4200K
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 3500)
        # Set Resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        print(f"Actual FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Actual Width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"Actual Height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"Actual Buffer: {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
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

"""
Raspberry Pi video streamer - sends video frames to computer and receives coordinates.

Usage:
    python -m connection.pi_streamer --camera usb0 --host 192.168.1.101

Run on the Raspberry Pi.
"""

import socket
import threading
import time
from typing import Callable, Optional
import cv2
import numpy as np

from . import config
from . import protocol


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
        if self.picam is not None:
            frame = self.picam.capture_array()
            # Convert RGB to BGR for OpenCV compatibility
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.cap is not None:
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def close(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.picam is not None:
            self.picam.stop()
            self.picam = None


class PiStreamer(protocol.ConnectionBase):
    """
    Video streamer for Raspberry Pi.

    Streams video frames to a computer and receives x,y coordinates back.
    """

    def __init__(self, host: str = config.COMPUTER_IP,
                 video_port: int = config.VIDEO_PORT,
                 coord_port: int = config.COORD_PORT):
        """
        Initialize the streamer.

        Args:
            host: Computer IP address to connect to
            video_port: Port for video streaming
            coord_port: Port for receiving coordinates
        """
        super().__init__()
        self.host = host
        self.video_port = video_port
        self.coord_port = coord_port
        self.camera: Optional[CameraCapture] = None
        self.frame_id = 0
        self.on_coordinates: Optional[Callable[[
            float, float, int, dict], None]] = None
        self._coord_thread: Optional[threading.Thread] = None

    def set_coordinate_callback(
            self, callback: Callable[[float, float, int, dict], None]):
        """
        Set callback for when coordinates are received.

        Args:
            callback: Function(x, y, frame_id, extra) called on coordinate receipt
        """
        self.on_coordinates = callback

    def connect(self) -> bool:
        """Connect to the computer. Returns True if successful."""
        try:
            # Connect video socket
            self.video_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.video_socket.settimeout(config.SOCKET_TIMEOUT)
            self.video_socket.connect((self.host, self.video_port))
            print(f"Connected video stream to {self.host}:{self.video_port}")

            # Connect coordinate socket
            self.coord_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.coord_socket.settimeout(config.SOCKET_TIMEOUT)
            self.coord_socket.connect((self.host, self.coord_port))
            print(
                f"Connected coordinate channel to {self.host}:{self.coord_port}")

            return True
        except socket.error as e:
            print(f"Connection failed: {e}")
            self.close()
            return False

    def start_camera(self, source: str = "usb0") -> bool:
        """
        Start camera capture.

        Args:
            source: Camera source ("usb0", "picamera0", etc.)

        Returns:
            True if camera opened successfully
        """
        self.camera = CameraCapture(
            source, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        if not self.camera.open():
            print(f"Failed to open camera: {source}")
            return False
        print(f"Camera opened: {source}")
        return True

    def _coordinate_receiver(self):
        """Background thread to receive coordinates."""
        while self.running:
            try:
                coords = protocol.recv_coordinates(self.coord_socket)
                if coords is None:
                    print(
                        "Coordinate connection lost. Disconnecting to find new client...")
                    self.running = False  # Signal main stream loop to stop
                    break
                if self.on_coordinates:
                    self.on_coordinates(
                        coords.get('x', 0),
                        coords.get('y', 0),
                        coords.get('frame_id', 0),
                        coords.get('extra', {})
                    )
            except socket.timeout:
                continue
            except Exception as e:
                print(
                    f"Coordinate receiver error: {e}. Disconnecting to find new client...")
                self.running = False  # Signal main stream loop to stop
                break

    def stream(self, max_fps: float = 30.0):
        """
        Start streaming video frames.

        Args:
            max_fps: Maximum frames per second to stream
        """
        if not self.camera:
            print("Camera not started")
            return
        if not self.video_socket:
            print("Not connected")
            return

        self.running = True
        frame_interval = 1.0 / max_fps

        # Start coordinate receiver thread
        self._coord_thread = threading.Thread(
            target=self._coordinate_receiver, daemon=True)
        self._coord_thread.start()

        print("Streaming started. Press Ctrl+C to stop.")
        try:
            while self.running:
                start_time = time.time()

                # Capture frame
                frame = self.camera.read()
                if frame is None:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue

                # Encode as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),
                                config.JPEG_QUALITY]
                _, encoded = cv2.imencode('.jpg', frame, encode_param)

                # Send frame
                if not protocol.send_frame(
                        self.video_socket,
                        encoded.tobytes(),
                        self.frame_id):
                    print("Failed to send frame")
                    break

                self.frame_id += 1

                # Rate limiting
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        except KeyboardInterrupt:
            print("\nStreaming stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop streaming and close connections."""
        self.running = False
        if self.camera:
            self.camera.close()
        self.close()

"""
Computer video receiver - receives video from Pi and sends coordinates back.

Usage:
    python -m connection.computer_receiver --host 0.0.0.0

Run on the computer that will process the video.
"""

import socket
import threading
import time
import argparse
from typing import Callable, Optional, Tuple
import cv2
import numpy as np

from . import config
from . import protocol


class ComputerReceiver(protocol.ConnectionBase):
    """
    Video receiver for computer.

    Receives video frames from Raspberry Pi and sends x,y coordinates back.
    """

    def __init__(self, host: str = "0.0.0.0",
                 video_port: int = config.VIDEO_PORT,
                 coord_port: int = config.COORD_PORT):
        """
        Initialize the receiver.

        Args:
            host: Host to bind to (0.0.0.0 for all interfaces)
            video_port: Port for video receiving
            coord_port: Port for sending coordinates
        """
        super().__init__()
        self.host = host
        self.video_port = video_port
        self.coord_port = coord_port

        self.video_server: Optional[socket.socket] = None
        self.coord_server: Optional[socket.socket] = None
        self.client_video: Optional[socket.socket] = None
        self.client_coord: Optional[socket.socket] = None

        self.on_frame: Optional[Callable[[np.ndarray, int], Optional[Tuple[float, float]]]] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id: int = 0
        self._lock = threading.Lock()

    def set_frame_callback(self, callback: Callable[[np.ndarray, int], Optional[Tuple[float, float]]]):
        """
        Set callback for processing frames.

        Args:
            callback: Function(frame, frame_id) -> (x, y) or None
                     Return coordinates to send back, or None to skip
        """
        self.on_frame = callback

    def start_servers(self) -> bool:
        """Start listening for connections. Returns True if successful."""
        try:
            # Video server
            self.video_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.video_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.video_server.bind((self.host, self.video_port))
            self.video_server.listen(1)
            print(f"Video server listening on {self.host}:{self.video_port}")

            # Coordinate server
            self.coord_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.coord_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.coord_server.bind((self.host, self.coord_port))
            self.coord_server.listen(1)
            print(f"Coordinate server listening on {self.host}:{self.coord_port}")

            return True
        except socket.error as e:
            print(f"Failed to start servers: {e}")
            self.close()
            return False

    def wait_for_connection(self) -> bool:
        """Wait for Pi to connect. Returns True when connected."""
        print("Waiting for Pi to connect...")
        try:
            self.video_server.settimeout(None)  # Block until connection
            self.client_video, addr = self.video_server.accept()
            print(f"Video connection from {addr}")

            self.coord_server.settimeout(config.SOCKET_TIMEOUT)
            self.client_coord, addr = self.coord_server.accept()
            print(f"Coordinate connection from {addr}")

            return True
        except socket.error as e:
            print(f"Connection error: {e}")
            return False

    def send_coordinates(self, x: float, y: float, frame_id: int = 0,
                         extra: dict = None) -> bool:
        """
        Send coordinates back to the Pi.

        Args:
            x: X coordinate
            y: Y coordinate
            frame_id: Corresponding frame ID
            extra: Optional extra data

        Returns:
            True if successful
        """
        if not self.client_coord:
            return False
        return protocol.send_coordinates(self.client_coord, x, y, frame_id, extra)

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        """Get the latest received frame. Thread-safe."""
        with self._lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_frame_id
        return None

    def receive_loop(self, show_video: bool = True, window_name: str = "Pi Camera"):
        """
        Main loop to receive and process frames.

        Args:
            show_video: Whether to display video in window
            window_name: OpenCV window name
        """
        if not self.client_video:
            print("No video connection")
            return

        self.running = True
        fps_start = time.time()
        frame_count = 0

        print("Receiving frames. Press 'q' to quit.")
        try:
            while self.running:
                # Receive frame
                result = protocol.recv_frame(self.client_video)
                if result is None:
                    print("Connection lost")
                    break

                frame_id, frame_data = result

                # Decode JPEG
                np_arr = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Store latest frame
                with self._lock:
                    self.latest_frame = frame.copy()
                    self.latest_frame_id = frame_id

                # Process frame with callback
                coords = None
                if self.on_frame:
                    coords = self.on_frame(frame, frame_id)

                # Send coordinates if provided
                if coords is not None:
                    x, y = coords
                    self.send_coordinates(x, y, frame_id)

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

                    if show_video:
                        cv2.setWindowTitle(window_name, f"{window_name} - {fps:.1f} FPS")

                # Display
                if show_video:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        cv2.imwrite(f"capture_{frame_id}.jpg", frame)
                        print(f"Saved capture_{frame_id}.jpg")

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            if show_video:
                cv2.destroyAllWindows()
            self.stop()

    def stop(self):
        """Stop receiving and close connections."""
        self.running = False
        if self.client_video:
            try:
                self.client_video.close()
            except:
                pass
        if self.client_coord:
            try:
                self.client_coord.close()
            except:
                pass
        if self.video_server:
            try:
                self.video_server.close()
            except:
                pass
        if self.coord_server:
            try:
                self.coord_server.close()
            except:
                pass


class FrameProcessor:
    """
    Modular frame processor for detecting objects and computing coordinates.

    Subclass this to implement custom processing logic.
    """

    def process(self, frame: np.ndarray, frame_id: int) -> Optional[Tuple[float, float]]:
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

    def __init__(self, window_name: str = "Pi Camera"):
        self.window_name = window_name
        self.click_coords: Optional[Tuple[float, float]] = None
        self._setup = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_coords = (float(x), float(y))
            print(f"Click: ({x}, {y})")

    def process(self, frame: np.ndarray, frame_id: int) -> Optional[Tuple[float, float]]:
        if not self._setup:
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            self._setup = True

        # Return and clear click coordinates
        coords = self.click_coords
        self.click_coords = None
        return coords


class CenterProcessor(FrameProcessor):
    """Processor that always returns the frame center."""

    def process(self, frame: np.ndarray, frame_id: int) -> Optional[Tuple[float, float]]:
        h, w = frame.shape[:2]
        return (w / 2.0, h / 2.0)


def main():
    parser = argparse.ArgumentParser(description="Computer Video Receiver")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--video-port", type=int, default=config.VIDEO_PORT,
                        help=f"Video port (default: {config.VIDEO_PORT})")
    parser.add_argument("--coord-port", type=int, default=config.COORD_PORT,
                        help=f"Coordinate port (default: {config.COORD_PORT})")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable video display")
    parser.add_argument("--mode", choices=["click", "center", "none"], default="click",
                        help="Coordinate mode: click (mouse clicks), center (frame center), none")
    args = parser.parse_args()

    # Create receiver
    receiver = ComputerReceiver(args.host, args.video_port, args.coord_port)

    # Set up processor
    window_name = "Pi Camera"
    if args.mode == "click":
        processor = ClickProcessor(window_name)
        receiver.set_frame_callback(processor.process)
    elif args.mode == "center":
        processor = CenterProcessor()
        receiver.set_frame_callback(processor.process)
    # mode == "none": no callback

    # Start servers
    if not receiver.start_servers():
        return

    # Main loop - wait for connections and process
    while True:
        if receiver.wait_for_connection():
            receiver.receive_loop(show_video=not args.no_display, window_name=window_name)
        print("Waiting for reconnection...")
        time.sleep(config.RECONNECT_DELAY)


if __name__ == "__main__":
    main()

"""
Computer video receiver - receives video from Pi and sends movement commands back.

This module handles the computer-side video streaming and movement command interface.
It receives video frames from the Raspberry Pi and sends back movement commands
consisting of motor coefficients and distance.

Usage:
    python -m connection.computer_receiver --host 0.0.0.0

Run on the computer that will process the video and send movement commands.
"""

import socket
import threading
import time
from typing import Callable, Optional, Tuple
import cv2
import numpy as np
import struct

from . import config
from . import protocol


class ComputerReceiver(protocol.ConnectionBase):
    """
    Video receiver and movement command sender for computer.

    Handles the computer-side of the video streaming and movement control system.
    Receives video frames from Raspberry Pi and sends back movement commands
    consisting of left/right motor coefficients and distance.
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

        self.on_frame: Optional[Callable[[np.ndarray, int],
                                Optional[Tuple[float, float, float]]]] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id: int = 0
        self._lock = threading.Lock()

    def set_frame_callback(self, callback: Callable[[
            np.ndarray, int], Optional[Tuple[float, float, float]]]):
        """
        Set callback for processing frames and generating movement commands.

        Args:
                callback: Function(frame, frame_id) -> (left_coef, right_coef, distance) or None
                                 Return movement command as a tuple of (left_coef, right_coef, distance),
                                 or None to skip sending movement command.
                                 - left_coef: Left motor coefficient (-1.0 to 1.0)
                                 - right_coef: Right motor coefficient (-1.0 to 1.0)
                                 - distance: Distance to move (in ticks)
        """
        self.on_frame = callback

    def start_servers(self) -> bool:
        """Start listening for connections. Returns True if successful."""
        try:
            # Video server
            self.video_server = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.video_server.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.video_server.bind((self.host, self.video_port))
            self.video_server.listen(1)
            print(f"Video server listening on {self.host}:{self.video_port}")

            # Coordinate server
            self.coord_server = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.coord_server.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.coord_server.bind((self.host, self.coord_port))
            self.coord_server.listen(1)
            print(
                f"Coordinate server listening on {self.host}:{self.coord_port}")

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

    def send_movement(
            self,
            left_coef: float,
            right_coef: float,
            distance: float) -> bool:
        """
        Send movement commands to the Pi.

        Args:
                left_coef: Left motor coefficient (-1.0 to 1.0)
                right_coef: Right motor coefficient (-1.0 to 1.0)
                distance: Distance to move (in meters)

        Returns:
                True if successful
        """
        if not self.client_coord:
            return False

        try:
            # Pack three 4-byte floats (12 bytes total)
            data = struct.pack('!fff', left_coef, right_coef, distance)
            return protocol.send_message(self.client_coord, data)
        except struct.error as e:
            print(f"Failed to pack movement command: {e}")
            return False

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        """Get the latest received frame. Thread-safe."""
        with self._lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_frame_id
        return None

    def receive_loop(self, show_video: bool = True,
                     window_name: str = "Pi Camera"):
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

                # Process frame with callback and get movement command
                if self.on_frame:
                    movement = self.on_frame(frame, frame_id)
                    if movement is not None:
                        left_coef, right_coef, distance = movement
                        self.send_movement(left_coef, right_coef, distance)

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

                    if show_video:
                        cv2.setWindowTitle(
                            window_name, f"{window_name} - {fps:.1f} FPS")

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
            except BaseException:
                pass
        if self.client_coord:
            try:
                self.client_coord.close()
            except BaseException:
                pass
        if self.video_server:
            try:
                self.video_server.close()
            except BaseException:
                pass
        if self.coord_server:
            try:
                self.coord_server.close()
            except BaseException:
                pass

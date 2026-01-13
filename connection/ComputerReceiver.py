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
import traceback

from . import config
from . import protocol


class ComputerReceiver():
    """
    Video receiver and movement command sender for computer.

    Handles the computer-side of the video streaming and movement control system.
    Receives video frames from Raspberry Pi and sends back movement commands
    consisting of left/right motor coefficients and distance.

    STARTING AND CLOSING CONNECTIONS IS NOT THREAD SAFE
    """

    def __init__(self, host: str = "0.0.0.0",
                 video_port: int = config.VIDEO_PORT,
                 command_port: int = config.COMMAND_PORT):
        """
        Initialize the receiver.

        Args:
            host: Host to bind to (0.0.0.0 for all interfaces)
            video_port: Port for video receiving
            command_port: Port for sending commands
        """
        super().__init__()
        self.host = host
        self.video_port = video_port
        self.command_port = command_port

        # Server sockets (listen for incoming connections)
        self.video_server_socket: Optional[socket.socket] = None
        self.command_server_socket: Optional[socket.socket] = None

        # Client sockets (active connections for data transfer)
        self.video_client_socket: Optional[socket.socket] = None
        self.command_client_socket: Optional[socket.socket] = None

        self.on_frame: Optional[Callable[[np.ndarray, int],
                                Optional[Tuple[float, float, float]]]] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id: int = 0
        self._lock = threading.Lock()

    def set_frame_callback(self, callback: Callable[[
            np.ndarray, int], Optional[Tuple[float, float, float]]]):
        """
        Set callback for processing frames and generating movement commands.
        THIS BLOCKS THE RECEIVING FRAMES LOOP

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
            # Set up video server socket
            self.video_server_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.video_server_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.video_server_socket.bind((self.host, self.video_port))
            self.video_server_socket.listen(1)
            print(f"Video server listening on {self.host}:{self.video_port}")

            # Set up movement command server socket
            self.command_server_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.command_server_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.command_server_socket.bind((self.host, self.command_port))
            self.command_server_socket.listen(1)
            print(
                f"Movement command server listening on {self.host}:{self.command_port}")

            return True
        except socket.error as e:
            print(f"Failed to start servers: {e}")
            return False

    def wait_for_connection(self) -> bool:
        """Wait for both video and movement command connections."""
        if self.video_server_socket is None or self.command_server_socket is None:
            print("Server sockets not initialized. Call start_servers() first.")
            return False

        print("Waiting for Pi to connect...")
        try:
            # Accept video connection
            self.video_client_socket, _ = self.video_server_socket.accept()
            print(
                f"Video connection from {self.video_client_socket.getpeername()}")

            # Accept movement command connection
            self.command_client_socket, _ = self.command_server_socket.accept()
            print(
                f"Movement command connection from {self.command_client_socket.getpeername()}")
            return True
        except socket.error as e:
            print(f"Connection error: {e}")
            return False

    def send_close(self) -> bool:
        """
        Send close command to the Pi to gracefully shut down the connection.

        Returns:
                True if successful
        """
        if not self.command_client_socket:
            return False

        return protocol.send_command(
            self.command_client_socket, config.MSG_TYPE_CLOSE, [])

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
        if not self.command_client_socket:
            return False

        return protocol.send_command(
            self.command_client_socket,
            config.MSG_TYPE_MOVEMENT,
            [left_coef, right_coef, distance])

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
        Runs until keyboard interrupt

        Args:
            show_video: Whether to display video in window
            window_name: OpenCV window name
        """
        if not self.video_client_socket:
            print("No video connection")
            return

        fps_start = time.time()
        frame_count = 0

        print("Receiving frames. Press 'q' to quit.")
        failed_frames: int = 0
        while True:
            try:
                # Receive frame
                result = protocol.recv_frame(self.video_client_socket)
                if result is None:
                    failed_frames += 1
                    if (failed_frames & (failed_frames - 1) == 0):
                        print(
                            f"Didn't Receive Frame (Connection Lost) {failed_frames}")
                    time.sleep(0.5)
                    continue
                else:
                    failed_frames = 0

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
                # TODO: might want to change this to also affect gripper
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
                self.close_client_connections()
                if show_video:
                    cv2.destroyAllWindows()
                raise KeyboardInterrupt
            except Exception as e:
                print(
                    f"Unexpected error in receive video loop: {type(e).__name__}: {e}")
                traceback.print_exc()

    def close_client_connections(self) -> None:
        """
        Close client connections safely.
        """
        # tell pi that we're closing the connection
        # so that it can look for a new one and see when we restart the code
        self.send_close()

        # Close client connections
        protocol.close_socket(self.video_client_socket)
        self.video_client_socket = None

        protocol.close_socket(self.command_client_socket)
        self.command_client_socket = None
        print("client connection stopped")

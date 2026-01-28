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
from typing import Callable, Optional
import cv2
import traceback
import config
from . import protocol
from connection.frame_info import FrameInfo
from connection.RemoteRobotCommander import RemoteRobotCommander
from profiler import Profiler


class ComputerReceiver:
    """
    Video receiver for computer.

    Handles the computer-side of the video streaming system.
    Receives video frames from Raspberry Pi.

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
        self.host = host
        self.video_port = video_port
        self.command_port = command_port

        # Server sockets (listen for incoming connections)
        self.video_server_socket: Optional[socket.socket] = None
        self.command_server_socket: Optional[socket.socket] = None

        # Client sockets (active connections for data transfer)
        self.video_client_socket: Optional[socket.socket] = None
        self.command_client_socket: Optional[socket.socket] = None

        # Remote commander for sending commands
        self.commander = RemoteRobotCommander()

        # takes in FrameInfo
        self.on_frame: Callable[[FrameInfo], None] = lambda _: None
        self._lock = threading.Lock()

        # Latest frame buffer for skip-ahead processing
        self.latest_frame: Optional[FrameInfo] = None
        self.frame_lock = threading.Lock()
        self.receive_thread: Optional[threading.Thread] = None
        self.should_stop = False
        self.frames_skipped = 0

        # Profiler for receive loop performance
        self.profiler = Profiler(verbose=False)

    def set_frame_callback(
            self, callback: Callable[[FrameInfo], None]):
        """
        Set callback for processing frames and generating movement commands.
        THIS BLOCKS THE RECEIVING FRAMES LOOP

        Args:
            callback: Function(frame_info) -> None
                - frame_info: FrameInfo object containing:
                    - frame_top: Top camera frame as numpy array
                    - frame_bottom: Bottom camera frame as numpy array
                    - frame_id: Frame sequence number
                    - x, y, theta: Robot position and orientation
                    - gripperHeight, gripperAngle, scooperAngle: Manipulator states
                    - distanceSensed: Distance sensor reading
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

            # Update commander's socket
            self.commander.set_socket(self.command_client_socket)
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
        return self._commander.close()

    def _receive_frames_thread(self):
        """Background thread that continuously receives frames and updates latest_frame."""
        failed_frames = 0
        while not self.should_stop:
            try:
                if not self.video_client_socket:
                    time.sleep(0.1)
                    continue

                # Receive frame info
                frame_info = protocol.recv_frame_info(self.video_client_socket)

                # Check for graceful disconnect
                if frame_info == 0:
                    print("Pi disconnected gracefully (receive thread)")
                    break

                if frame_info is None:
                    failed_frames += 1
                    if (failed_frames & (failed_frames - 1) == 0):
                        print(
                            f"Didn't Receive Frame (Connection Lost) {failed_frames}")
                    time.sleep(0.5)
                    continue
                else:
                    failed_frames = 0

                # Update latest frame
                with self.frame_lock:
                    if self.latest_frame is not None:
                        self.frames_skipped += 1
                    self.latest_frame = frame_info

            except Exception as e:
                if not self.should_stop:
                    print(f"Error in receive thread: {e}")
                    traceback.print_exc()
                time.sleep(0.5)

    def receive_loop(self, show_video: bool = True,
                     window_name_top: str = "Top Camera",
                     window_name_bottom: str = "Bottom Camera"):
        """
        Main loop to receive and process frames.
        Always processes the latest frame, skipping old frames if processing is slow.
        Runs until keyboard interrupt

        Args:
            show_video: Whether to display video in windows
            window_name_top: OpenCV window name for top camera
            window_name_bottom: OpenCV window name for bottom camera
        """
        if not self.video_client_socket:
            print("No video connection")
            return

        # Start background thread to receive frames
        self.should_stop = False
        self.receive_thread = threading.Thread(
            target=self._receive_frames_thread, daemon=True)
        self.receive_thread.start()

        fps_start = time.time()
        frame_count = 0
        last_frame_id = -1
        last_skipped_count = 0

        print("Receiving frames. Press 'q' to quit.")
        print("Press 'p' to save profiler data.")

        try:
            while True:
                try:
                    self.profiler.start_frame()

                    # Get the latest frame
                    frame_info = None
                    with self.frame_lock:
                        if self.latest_frame is not None:
                            frame_info = self.latest_frame
                            self.latest_frame = None  # Clear it so we know when a new one arrives

                    # If no new frame available, wait a bit
                    if frame_info is None:
                        time.sleep(0.01)
                        continue

                    # Check if we skipped frames
                    if self.frames_skipped > last_skipped_count:
                        frames_skipped_this_cycle = self.frames_skipped - last_skipped_count
                        # print(
                        # f"Skipped {frames_skipped_this_cycle} frames (total:
                        # {self.frames_skipped})")
                        last_skipped_count = self.frames_skipped

                    last_frame_id = frame_info.frame_id

                    self.profiler.record("get_latest_frame")

                    # Process frame with callback
                    try:
                        self.on_frame(frame_info)
                        self.profiler.record("frame_callback")
                    except Exception as e:
                        print("Error in frame callback")
                        print(e)
                        traceback.print_exc()

                    # Calculate FPS
                    frame_count += 1
                    elapsed = time.time() - fps_start
                    if elapsed >= 1.0:
                        fps = frame_count / elapsed
                        frame_count = 0
                        fps_start = time.time()

                        if show_video:
                            cv2.setWindowTitle(
                                window_name_top, f"{window_name_top} - {fps:.1f} FPS")

                    # Display both frames
                    if show_video:
                        cv2.imshow(window_name_top, frame_info.frame_top)
                        cv2.imshow(window_name_bottom, frame_info.frame_bottom)
                        self.profiler.record("imshow")
                        key = cv2.waitKey(1) & 0xFF
                        self.profiler.record("waitKey")

                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            cv2.imwrite(
                                f"capture_top_{frame_info.frame_id}.jpg",
                                frame_info.frame_top)
                            cv2.imwrite(
                                f"capture_bottom_{frame_info.frame_id}.jpg",
                                frame_info.frame_bottom)
                            print(
                                f"Saved capture_top_{frame_info.frame_id}.jpg and capture_bottom_{frame_info.frame_id}.jpg")
                        elif key == ord('p'):
                            print("Saving profiler data...")
                            self.profiler.save_profile()
                            print("Profiler data saved!")

                    self.profiler.end_frame()
                except KeyboardInterrupt:
                    print("\nStopped by user")
                    self.should_stop = True
                    if self.receive_thread:
                        self.receive_thread.join(timeout=2.0)
                    self.close_client_connections()
                    if show_video:
                        cv2.destroyAllWindows()
                    raise KeyboardInterrupt
                except Exception as e:
                    print(
                        f"Unexpected error in receive video loop: {type(e).__name__}: {e}")
                    traceback.print_exc()
        finally:
            # Ensure thread is stopped
            self.should_stop = True
            if self.receive_thread:
                self.receive_thread.join(timeout=2.0)

    def close_client_connections(self) -> None:
        """
        Close client connections safely.
        """
        # tell pi that we're closing the connection
        # so that it can look for a new one and see when we restart the code
        self.commander.close()

        # Close client connections
        protocol.close_socket(self.video_client_socket)
        self.video_client_socket = None

        protocol.close_socket(self.command_client_socket)
        self.commander.set_socket(None)
        self.command_client_socket = None
        print("client connection stopped")

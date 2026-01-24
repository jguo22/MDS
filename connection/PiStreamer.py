"""
Raspberry Pi video streamer - sends video frames to computer and receives movement commands.

This module handles the Raspberry Pi side of the video streaming and movement control system.
It streams video frames to a connected computer and receives movement commands consisting
of left/right motor coefficients and distance.

IMPORTANT: PiStreamer is designed for single-use. Each instance handles one connection lifecycle.
For reconnection, create a new PiStreamer instance while reusing the same camera.

Usage example:
    from connection.CameraCapture import CameraCapture
    from connection.PiStreamer import PiStreamer

    camera = CameraCapture("usb0", 640, 480)
    camera.open()

    while True:
        streamer = PiStreamer(camera=camera, host="192.168.1.101")
        streamer.set_movement_callback(handle_movement)
        if streamer.connect():
            streamer.stream()  # Blocks until disconnected
        time.sleep(2)  # Brief pause before reconnecting

Run on the Raspberry Pi that will stream video and receive movement commands.
"""

import socket
import threading
import time
from typing import Callable, Optional
import traceback

from IMUWrapper import IMUWrapper
from RavenWrapper import RavenWrapper
from connection import message_types
from connection.frame_info import FrameInfo
from profiler import Profiler

import config
from . import protocol
from .CameraCapture import CameraCapture


class PiStreamer():
    """
    Video streamer and movement command receiver for Raspberry Pi.

    Handles the Raspberry Pi side of the video streaming and movement control system.
    Streams video frames to a connected computer and receives movement commands
    consisting of left/right motor coefficients and distance.
    """

    def __init__(self, camera_top: CameraCapture,
                 camera_bottom: CameraCapture,
                 ravenWrapper: RavenWrapper,
                 imuWrapper: IMUWrapper,
                 host: str = config.COMPUTER_IP,
                 video_port: int = config.VIDEO_PORT,
                 command_port: int = config.COMMAND_PORT):
        """
        Initialize the streamer for a single-use connection.

        Args:
            camera_top: Top CameraCapture instance (managed externally)
            camera_bottom: Bottom CameraCapture instance (managed externally)
            raven: Raven motor controller instance (for odometry)
            imu_wrapper: IMUWrapper instance (for heading)
            host: Computer IP address to connect to
            video_port: Port for video streaming
            command_port: Port for receiving commands
        """

        self.camera_top = camera_top
        self.camera_bottom = camera_bottom
        self.ravenWrapper = ravenWrapper
        self.imu_wrapper = imuWrapper
        self.host = host
        self.video_port = video_port
        self.command_port = command_port
        # Client sockets for communication
        self.video_client_socket: Optional[socket.socket] = None
        self.command_client_socket: Optional[socket.socket] = None
        self.frame_id = 0
        self.running = False

        # Sensor value getters (set these to get real sensor data)
        self.get_gripper_height: Callable[[], float] = lambda: 0.0
        self.get_gripper_angle: Callable[[], float] = lambda: 0.0
        self.get_scooper_angle: Callable[[], float] = lambda: 0.0
        self.get_distance_sensed: Callable[[], float] = lambda: 0.0

        # movement callback blocks the command receiving thread
        # takes in msg_type and args and does the command
        self.command_callback: Callable[[
            int, list[float]], None] = (lambda _, __: None)

        self._command_receiver_thread: Optional[threading.Thread] = None

        # Profiler for stream performance
        self.profiler = Profiler()

    def set_command_callback(
            self, callback: Callable[[int, list[float]], None]):
        """
        Set callback for when movement commands are received.
        NOTE: movement callback blocks the command receiver thread

        Args:
            callback: Function(msg_type, args) called on command receipt
        """
        self.command_callback = callback

    def set_sensor_callbacks(
            self,
            get_gripper_height: Optional[Callable[[], float]] = None,
            get_gripper_angle: Optional[Callable[[], float]] = None,
            get_scooper_angle: Optional[Callable[[], float]] = None,
            get_distance_sensed: Optional[Callable[[], float]] = None):
        """
        Set callbacks for sensor data retrieval.

        Args:
            get_gripper_height: Function returning gripper height in mm
            get_gripper_angle: Function returning gripper angle in radians
            get_scooper_angle: Function returning scooper angle in radians
            get_distance_sensed: Function returning distance sensor reading in mm
        """
        if get_gripper_height is not None:
            self.get_gripper_height = get_gripper_height
        if get_gripper_angle is not None:
            self.get_gripper_angle = get_gripper_angle
        if get_scooper_angle is not None:
            self.get_scooper_angle = get_scooper_angle
        if get_distance_sensed is not None:
            self.get_distance_sensed = get_distance_sensed

    def connect(self) -> bool:
        """
        Connect to the computer. Single attempt only.
        For reconnection, create a new PiStreamer instance.

        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Connect video client socket
            self.video_client_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.video_client_socket.settimeout(config.SOCKET_TIMEOUT)
            self.video_client_socket.connect((self.host, self.video_port))
            print(
                f"Connected video stream to {self.host}:{self.video_port}")

            # Connect to command server
            self.command_client_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.command_client_socket.settimeout(config.SOCKET_TIMEOUT)
            self.command_client_socket.connect(
                (self.host, self.command_port))
            print(
                f"Connected to command server at {
                    self.host}:{
                    self.command_port}")

            # Start command receiver thread
            self.running = True
            self._command_receiver_thread = threading.Thread(
                target=self._command_receiver, daemon=True)
            self._command_receiver_thread.start()
            print("Command receiver thread started")

            return True

        except Exception as e:
            print(f"Unexpected error during connection: {e}")
            traceback.print_exc()
            self.stop()
            return False

    def _command_receiver(self):
        """Background thread to receive and process command messages."""
        while self.running:
            try:
                if not self.command_client_socket:
                    time.sleep(0.1)
                    continue

                # Receive command using protocol helper
                result = protocol.recv_command(self.command_client_socket)
                if result is None:
                    # Failed to receive/parse command - could be malformed data, timeout, etc.
                    # Just log and continue - don't treat as shutdown signal
                    print("Failure receiving/processing command")
                    time.sleep(0.1)
                    continue

                msg_type, args = result

                # Handle close command (type 0) - explicit shutdown signal
                if msg_type == message_types.CLOSE:
                    print("Received close command from computer")
                    self.stop()
                    break
                else:
                    try:
                        self.command_callback(msg_type, args)
                    except Exception as e:
                        print(
                            f"Error in movement callback: {
                                type(e).__name__}: {e}")
                        traceback.print_exc()

            except Exception as e:
                print(
                    f"Unexpected error in command receiver: {
                        type(e).__name__}: {e}")
                traceback.print_exc()
                # don't shut down entire robot from just one frame of failure
                continue

        print("Command receiver thread exiting")

    def stream(self, max_fps: float = config.FPS):
        """
        Start streaming video frames. Blocks until connection is lost or stopped.

        Args:
            max_fps: Maximum frames per second to stream (default: config.DEFAULT_MAX_FPS)
        """
        if self.camera_top is None or not self.camera_top.is_open():
            print("Cannot start streaming: top camera not available")
            return
        if self.camera_bottom is None or not self.camera_bottom.is_open():
            print("Cannot start streaming: bottom camera not available")
            return
        if not self.video_client_socket:
            print("Not connected to video server")
            return

        frame_interval = 1.0 / max_fps

        print("Streaming started. Press Ctrl+C to stop.")
        print(
            f"Target FPS: {max_fps} (frame interval: {
                frame_interval *
                1000:.1f}ms)")

        while self.running:
            try:
                self.profiler.start_frame()
                start_time = time.time()

                # Capture frames from both cameras
                frame_top = self.camera_top.read()
                self.profiler.record("camera_read_top")
                frame_bottom = self.camera_bottom.read()
                self.profiler.record("camera_read_bot")
                if frame_top is None or frame_bottom is None:
                    print("failed to read frame from cameras")
                    time.sleep(0.1)
                    continue
                self.profiler.record("camera_read")

                # Get current robot pose
                x, y = self.ravenWrapper.get_odometry()
                theta = self.imu_wrapper.get_heading()
                self.profiler.record("get_pose")

                # Get sensor data
                gripper_height = self.get_gripper_height()
                gripper_angle = self.get_gripper_angle()
                scooper_angle = self.get_scooper_angle()
                distance_sensed = self.get_distance_sensed()
                self.profiler.record("get_sensors")

                # Create FrameInfo object
                frame_info = FrameInfo(
                    frame_top=frame_top,
                    frame_bottom=frame_bottom,
                    frame_id=self.frame_id,
                    x=x,
                    y=y,
                    theta=theta,
                    gripperHeight=gripper_height,
                    gripperAngle=gripper_angle,
                    scooperAngle=scooper_angle,
                    distanceSensed=distance_sensed
                )

                self.profiler.record("send_frame0")
                # Send frame info
                if not protocol.send_frame_info(
                        self.video_client_socket,
                        frame_info,
                        config.JPEG_QUALITY):
                    print("Failed to send frame. Continuing...")
                    continue
                self.profiler.record("send_frame")

                self.frame_id += 1

                # Rate limiting
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    sleep_time = frame_interval - elapsed
                    time.sleep(sleep_time)
                self.profiler.record("rate_limit_sleep")

                self.profiler.end_frame()
            except KeyboardInterrupt:
                print("\nStreaming stopped by user")
                self.stop()
                raise KeyboardInterrupt
            except Exception as e:
                print(f"Unexpected streaming error: {e}")
                traceback.print_exc()

    def stop(self):
        """Stop streaming and close socket connections."""
        self.running = False
        # Send graceful disconnect signal before closing
        if self.video_client_socket:
            protocol.send_disconnect_from_pi(self.video_client_socket)
        protocol.close_socket(self.video_client_socket)
        self.video_client_socket = None
        protocol.close_socket(self.command_client_socket)
        self.command_client_socket = None
        print("PiStreamer stopped")

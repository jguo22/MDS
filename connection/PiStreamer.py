"""
Raspberry Pi video streamer - sends video frames to computer and receives movement commands.

This module handles the Raspberry Pi side of the video streaming and movement control system.
It streams video frames to a connected computer and receives movement commands consisting
of left/right motor coefficients and distance.

Usage:
    python -m connection.pi_streamer --camera usb0 --host 192.168.1.101

Run on the Raspberry Pi that will stream video and receive movement commands.
"""

import socket
import threading
import time
import cv2
from typing import Callable, Optional

from . import config
from . import protocol
from .CameraCapture import CameraCapture


class PiStreamer(protocol.ConnectionBase):
    """
    Video streamer and movement command receiver for Raspberry Pi.

    Handles the Raspberry Pi side of the video streaming and movement control system.
    Streams video frames to a connected computer and receives movement commands
    consisting of left/right motor coefficients and distance.
    """

    def __init__(self, host: str = config.COMPUTER_IP,
                 video_port: int = config.VIDEO_PORT,
                 movement_port: int = config.MOVEMENT_PORT):
        """
        Initialize the streamer.

        Args:
            host: Computer IP address to connect to
            video_port: Port for video streaming
            movement_port: Port for receiving movement commands
        """
        super().__init__()
        self.host = host
        self.video_port = video_port
        self.movement_port = movement_port
        # Client sockets for communication
        self.video_client_socket: Optional[socket.socket] = None
        # Store source for reopening
        self.movement_client_socket: Optional[socket.socket] = None
        self.frame_id = 0
        self.movement_callback: Optional[Callable[[
            float, float, float], None]] = None

    def set_movement_callback(
            self, callback: Callable[[float, float, float], None]):
        """
        Set callback for when movement commands are received.

        Args:
            callback: Function(x, y, frame_id, extra) called on movement command receipt
        """
        self.movement_callback = callback

    def connect(
            self,
            max_attempts: int = 5,
            initial_delay: float = 1.0) -> bool:
        """
        Connect to the computer. Returns True if successful.

        Args:
            max_attempts: Maximum number of connection attempts
            initial_delay: Initial delay between attempts in seconds (will double after each attempt)

        Returns:
            bool: True if connection was successful, False otherwise
        """
        self.running = True
        attempt = 0
        delay = initial_delay

        while True:
            attempt += 1
            try:
                # Connect video client socket
                self.video_client_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.video_client_socket.settimeout(config.SOCKET_TIMEOUT)
                self.video_client_socket.connect((self.host, self.video_port))
                print(
                    f"Connected video stream to {self.host}:{self.video_port}")

                # Connect to movement command server
                self.movement_client_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.movement_client_socket.settimeout(config.SOCKET_TIMEOUT)
                self.movement_client_socket.connect(
                    (self.host, self.movement_port))
                print(
                    f"Connected to movement command server at {self.host}:{self.movement_port}")

                # Start movement receiver thread
                if not hasattr(
                        self,
                        '_movement_thread') or not self._movement_thread.is_alive():
                    self._movement_thread = threading.Thread(
                        target=self._movement_receiver, daemon=True)
                    self._movement_thread.start()
                    print("Movement receiver thread started")

                return True

            except socket.error as e:
                if attempt >= max_attempts:
                    print(
                        f"Failed to connect after {max_attempts} attempts: {e}")
                    self.close()
                    return False

                print(
                    f"Connection attempt {attempt} failed: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 10)  # Exponential backoff with max 10s

            except Exception as e:
                print(f"Unexpected error during connection: {e}")
                self.close()
                return False

        return False

    def start_camera(self, source: str = "usb0") -> bool:
        """
        Start camera capture.

        Args:
            source: Camera source ("usb0", "picamera0", etc.)

        Returns:
            True if camera opened successfully
        """
        self._camera_source = source  # Store for reopening
        self.camera = CameraCapture(
            source, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        if self.camera.open():
            print(f"Camera opened: {source}")
            return True
        else:
            print(f"Failed to open camera: {source}")
            return False

    def _ensure_camera_open(self) -> bool:
        """Ensure camera is open, reopen if needed. Returns True if camera is ready."""
        if self.camera is None:
            if self._camera_source is None:
                print("No camera source configured")
                return False
            return self.start_camera(self._camera_source)

        if not self.camera.is_open():
            print("Camera closed, attempting to reopen...")
            if not self.camera.reopen():
                print("Failed to reopen camera")
                return False
            print("Camera reopened successfully")
        return True

    def _movement_receiver(self):
        """Background thread to receive and process movement commands."""
        while self.running:
            try:
                if not self.movement_client_socket:
                    raise Exception("No Movement Socket")

                # Receive movement command using protocol helper
                movement = protocol.recv_movement(
                    self.movement_client_socket)
                if movement is None:
                    continue

                left_coef = movement['left_coef']
                right_coef = movement['right_coef']
                distance = movement['distance']

                print(
                    f"Received movement command - L: {left_coef:.2f}, R: {right_coef:.2f}, D: {distance:.2f}")

                if self.movement_callback:
                    try:
                        self.movement_callback(
                            left_coef, right_coef, distance)
                    except Exception as e:
                        print(
                            f"Error in movement callback: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()

            except (socket.error) as e:
                print(f"Movement socket error: {e}")
                time.sleep(1)

            except Exception as e:
                print(
                    f"Unexpected error in movement receiver: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

    def stream(self, max_fps: float = 30.0):
        """
        Start streaming video frames.

        Args:
            max_fps: Maximum frames per second to stream
        """
        # Ensure camera is open before streaming
        if not self._ensure_camera_open():
            print("Cannot start streaming: camera not available")
            return
        if not self.video_client_socket:
            print("Not connected to video server")
            return

        self.running = True
        frame_interval = 1.0 / max_fps

        print("Streaming started. Press Ctrl+C to stop.")
        consecutive_failures = 0
        max_consecutive_failures = 10

        try:
            while self.running:
                start_time = time.time()

                # Capture frame
                frame = self.camera.read()
                if frame is None:
                    consecutive_failures += 1
                    # Try to reopen camera after a few failures
                    if consecutive_failures == 5:
                        print("Multiple frame failures, attempting camera reopen...")
                        if self.camera.reopen():
                            print("Camera reopened, retrying...")
                            consecutive_failures = 0
                            continue
                    if consecutive_failures >= max_consecutive_failures:
                        print(
                            f"Too many consecutive frame capture failures ({max_consecutive_failures}). Disconnecting...")
                        break
                    time.sleep(0.1)
                    continue

                # Reset failure counter on successful capture
                consecutive_failures = 0

                # Encode as JPEG
                try:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),
                                    config.JPEG_QUALITY]
                    success, encoded = cv2.imencode(
                        '.jpg', frame, encode_param)
                    if not success:
                        print("Failed to encode frame")
                        continue
                except Exception as e:
                    print(f"Frame encoding error: {e}")
                    continue

                # Send frame
                try:
                    if not protocol.send_frame(
                            self.video_client_socket,
                            encoded.tobytes(),
                            self.frame_id):
                        print("Failed to send frame. Disconnecting...")
                        break
                except (socket.error, OSError, BrokenPipeError) as e:
                    print(
                        f"Socket error while sending frame: {e}. Disconnecting...")
                    break
                except Exception as e:
                    print(
                        f"Unexpected error sending frame: {e}. Disconnecting...")
                    break

                self.frame_id += 1

                # Rate limiting
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

        except KeyboardInterrupt:
            print("\nStreaming stopped by user")
        except Exception as e:
            print(f"Unexpected streaming error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop streaming and close socket connections. Camera stays open for reconnect."""
        self.running = False
        self.close()  # Uses parent's safe close method (sockets only)

    def shutdown(self):
        """Full shutdown - close everything including camera."""
        self.stop()
        if self.camera:
            self.camera.close()
            self.camera = None

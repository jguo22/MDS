"""
Connection module for video streaming between Raspberry Pi and computer.

Components:
- PiStreamer: Run on Pi to stream video and receive movement
- ComputerReceiver: Run on computer to receive video and send movement
- CameraCapture: Modular camera capture (USB, PiCamera)
- FrameProcessor: Base class for custom frame processing

Usage:
    # On Raspberry Pi:
    python -m connection.pi_streamer --host <computer_ip> --camera usb0

    # On Computer:
    python -m connection.computer_receiver

    # As a library:
    from connection.pi_streamer import PiStreamer
    from connection.computer_receiver import ComputerReceiver

See config.py for configuration options.
"""

# Only export config - submodules are imported on demand to avoid circular
# import warnings
from .config import (
    PI_IP,
    COMPUTER_IP,
    VIDEO_PORT,
    COMMAND_PORT,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    JPEG_QUALITY,
    DEFAULT_MAX_FPS,
    RECONNECT_DELAY,
)

__all__ = [
    'PI_IP',
    'COMPUTER_IP',
    'VIDEO_PORT',
    'COMMAND_PORT',
    'FRAME_WIDTH',
    'FRAME_HEIGHT',
    'JPEG_QUALITY',
    'DEFAULT_MAX_FPS',
    'RECONNECT_DELAY',
]

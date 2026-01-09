"""
Connection module for video streaming between Raspberry Pi and computer.

Components:
- PiStreamer: Run on Pi to stream video and receive coordinates
- ComputerReceiver: Run on computer to receive video and send coordinates
- CameraCapture: Modular camera capture (USB, PiCamera)
- FrameProcessor: Base class for custom frame processing

Usage:
    # On Raspberry Pi:
    python -m connection.pi_streamer --host <computer_ip> --camera usb0

    # On Computer:
    python -m connection.computer_receiver

See config.py for configuration options.
"""

from .config import (
    PI_IP,
    COMPUTER_IP,
    VIDEO_PORT,
    COORD_PORT,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    JPEG_QUALITY,
)

from .protocol import (
    send_message,
    recv_message,
    send_frame,
    recv_frame,
    send_coordinates,
    recv_coordinates,
    ConnectionBase,
)

from .pi_streamer import (
    PiStreamer,
    CameraCapture,
)

from .computer_receiver import (
    ComputerReceiver,
    FrameProcessor,
    ClickProcessor,
    CenterProcessor,
)

__all__ = [
    # Config
    'PI_IP',
    'COMPUTER_IP',
    'VIDEO_PORT',
    'COORD_PORT',
    'FRAME_WIDTH',
    'FRAME_HEIGHT',
    'JPEG_QUALITY',
    # Protocol
    'send_message',
    'recv_message',
    'send_frame',
    'recv_frame',
    'send_coordinates',
    'recv_coordinates',
    'ConnectionBase',
    # Pi
    'PiStreamer',
    'CameraCapture',
    # Computer
    'ComputerReceiver',
    'FrameProcessor',
    'ClickProcessor',
    'CenterProcessor',
]

"""
Protocol utilities for reliable message passing over TCP sockets.
"""

import struct
import json
import socket
from typing import Optional, Tuple, Any

from . import config


def send_message(sock: socket.socket, data: bytes) -> bool:
    """
    Send a message with a length header.

    Args:
        sock: Socket to send on
        data: Raw bytes to send

    Returns:
        True if successful, False otherwise
    """
    try:
        # Pack message length as 8-byte unsigned long
        header = struct.pack('!Q', len(data))
        sock.sendall(header + data)
        return True
    except (socket.error, BrokenPipeError):
        return False


def recv_message(sock: socket.socket) -> Optional[bytes]:
    """
    Receive a message with a length header.

    Args:
        sock: Socket to receive from

    Returns:
        Message bytes or None if failed
    """
    try:
        # Receive header
        header = _recv_exact(sock, config.HEADER_SIZE)
        if header is None:
            return None

        # Unpack message length
        msg_len = struct.unpack('!Q', header)[0]

        # Receive message body
        return _recv_exact(sock, msg_len)
    except (socket.error, struct.error):
        return None


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """
    Receive exactly n bytes from socket.

    Args:
        sock: Socket to receive from
        n: Number of bytes to receive

    Returns:
        Bytes received or None if connection closed
    """
    data = b''
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), config.BUFFER_SIZE))
        if not chunk:
            return None
        data += chunk
    return data


def send_frame(sock: socket.socket, frame_data: bytes, frame_id: int = 0) -> bool:
    """
    Send a video frame with metadata.

    Args:
        sock: Socket to send on
        frame_data: JPEG-encoded frame bytes
        frame_id: Frame sequence number

    Returns:
        True if successful
    """
    # Create frame packet: 4-byte frame_id + frame data
    packet = struct.pack('!I', frame_id) + frame_data
    return send_message(sock, packet)


def recv_frame(sock: socket.socket) -> Optional[Tuple[int, bytes]]:
    """
    Receive a video frame with metadata.

    Args:
        sock: Socket to receive from

    Returns:
        Tuple of (frame_id, frame_data) or None if failed
    """
    data = recv_message(sock)
    if data is None or len(data) < 4:
        return None

    frame_id = struct.unpack('!I', data[:4])[0]
    frame_data = data[4:]
    return frame_id, frame_data


def send_coordinates(sock: socket.socket, x: float, y: float,
                     frame_id: int = 0, extra: dict = None) -> bool:
    """
    Send x,y coordinates back to the Pi.

    Args:
        sock: Socket to send on
        x: X coordinate
        y: Y coordinate
        frame_id: Corresponding frame ID
        extra: Optional extra data dict

    Returns:
        True if successful
    """
    payload = {
        'x': x,
        'y': y,
        'frame_id': frame_id,
    }
    if extra:
        payload['extra'] = extra

    data = json.dumps(payload).encode('utf-8')
    return send_message(sock, data)


def recv_coordinates(sock: socket.socket) -> Optional[dict]:
    """
    Receive x,y coordinates.

    Args:
        sock: Socket to receive from

    Returns:
        Dict with 'x', 'y', 'frame_id' keys or None if failed
    """
    data = recv_message(sock)
    if data is None:
        return None

    try:
        return json.loads(data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


class ConnectionBase:
    """Base class for connection handling."""

    def __init__(self):
        self.video_socket: Optional[socket.socket] = None
        self.coord_socket: Optional[socket.socket] = None
        self.running = False

    def close(self):
        """Close all sockets."""
        self.running = False
        if self.video_socket:
            try:
                self.video_socket.close()
            except:
                pass
            self.video_socket = None
        if self.coord_socket:
            try:
                self.coord_socket.close()
            except:
                pass
            self.coord_socket = None

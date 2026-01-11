"""
Protocol utilities for reliable message passing over TCP sockets.
"""

import struct
import socket
from typing import Optional, Tuple

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


def send_frame(
        sock: socket.socket,
        frame_data: bytes,
        frame_id: int = 0) -> bool:
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


def send_movement(sock: socket.socket, left_coef: float, right_coef: float,
                  distance: float) -> bool:
    """
    Send movement commands to the Pi.

    Args:
        sock: Socket to send on
        left_coef: Left motor coefficient (-1.0 to 1.0)
        right_coef: Right motor coefficient (-1.0 to 1.0)
        distance: Distance to move (in ticks)

    Returns:
        True if successful
    """
    try:
        # Pack three 4-byte floats (12 bytes total)
        data = struct.pack('!fff', left_coef, right_coef, distance)
        return send_message(sock, data)
    except struct.error as e:
        print(f"Failed to pack movement command: {e}")
        return False


def recv_movement(sock: socket.socket) -> Optional[dict]:
    """
    Receive movement commands.

    Args:
        sock: Socket to receive from

    Returns:
        Dict with 'left_coef', 'right_coef', 'distance' or None if failed
    """
    # Each float is 4 bytes, so we expect 12 bytes total
    data = _recv_exact(sock, 12)
    if data is None or len(data) != 12:
        return None

    try:
        # Unpack three 4-byte floats
        left_coef, right_coef, distance = struct.unpack('!fff', data)
        return {
            'left_coef': left_coef,
            'right_coef': right_coef,
            'distance': distance
        }
    except struct.error as e:
        print(f"Failed to unpack movement command: {e}")
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
            except BaseException:
                pass
            self.video_socket = None
        if self.coord_socket:
            try:
                self.coord_socket.close()
            except BaseException:
                pass
            self.coord_socket = None

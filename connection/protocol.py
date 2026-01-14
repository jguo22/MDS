"""
Protocol utilities for reliable message passing over TCP sockets.
"""

import struct
import socket
from typing import Optional, Tuple

from . import config


def close_socket(sock: Optional[socket.socket]) -> None:
    """
    Safely close a socket with proper error handling.

    Args:
        sock: Socket to close, or None
    """
    if sock is None:
        return

    try:
        # Shutdown socket to stop any ongoing operations
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, socket.error):
            # Socket may already be closed or not connected
            pass

        # Close the socket
        sock.close()
    except Exception as e:
        # Catch any other unexpected errors during close
        print(f"Error closing socket: {e}")


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


def send_command(
        sock: socket.socket,
        msg_type: int,
        args: list[float]) -> bool:
    """
    Send a generic command message.

    Args:
        sock: Socket to send on
        msg_type: Message type identifier (0-255)
        args: List of float arguments

    Returns:
        True if successful
    """
    try:
        # Validate message type
        if msg_type not in [msgType.value for msgType in config.MessageType]:
            print(f"Unknown message type: {msg_type}")
            return False

        # Pack: 1-byte message type + N 4-byte floats
        data = struct.pack('!B', msg_type) + \
            struct.pack(f'!{len(args)}f', *args)
        return send_message(sock, data)
    except struct.error as e:
        print(f"Failed to pack command: {e}")
        return False


def recv_command(sock: socket.socket) -> Optional[Tuple[int, list[float]]]:
    """
    Receive a generic command message.

    Args:
        sock: Socket to receive from

    Returns:
        Tuple of (msg_type, args) where args is a list of floats, or None if failed
    """
    # Receive message with length header
    data = recv_message(sock)
    print(f'data is {data}')
    if data is None or len(data) < 1:
        return None

    try:
        # Unpack message type (1 byte)
        msg_type: int = struct.unpack('!B', data[:1])[0]

        # Validate message type
        if msg_type not in [msgType.value for msgType in config.MessageType]:
            print(f"Unknown message type: {msg_type}")
            return None

        # 1 byte type + N floats (4 bytes each) = length of data
        arg_count = (len(data) - 1) // 4

        # Unpack float arguments
        args = list(struct.unpack(f'!{arg_count}f', data[1:]))

        return msg_type, args
    except struct.error as e:
        print(f"Failed to unpack command: {e}")
        return None

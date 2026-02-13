"""
Protocol utilities for reliable message passing over TCP sockets.
"""

import struct
import socket
from typing import Optional, Tuple, Literal
from connection import message_types
from connection.frame_info import FrameInfo
import config
import cv2
import numpy as np


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
        frame_id: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        camera_angle: float = 0.0) -> bool:
    """
    Send a video frame with metadata and robot pose.

    Args:
        sock: Socket to send on
        frame_data: JPEG-encoded frame bytes
        frame_id: Frame sequence number
        x: Robot x position in mm (world coordinates)
        y: Robot y position in mm (world coordinates)
        theta: Robot orientation in radians
        camera_angle: Camera servo angle in radians

    Returns:
        True if successful
    """
    # Create frame packet: 4-byte frame_id + 4 floats (x, y, theta,
    # camera_angle) + frame data
    packet = struct.pack('!I', frame_id) + \
        struct.pack('!ffff', x, y, theta, camera_angle) + frame_data
    return send_message(sock, packet)


def send_disconnect_from_pi(sock: socket.socket) -> bool:
    """
    Send graceful disconnect signal from Pi (single 0 byte).

    Args:
        sock: Socket to send on

    Returns:
        True if successful
    """
    return send_message(sock, b'\x00')


def recv_frame(
        sock: socket.socket) -> Optional[Tuple[bytes, int, float, float, float, float] | Literal[0]]:
    """
    Receive a video frame with metadata and robot pose.

    Args:
        sock: Socket to receive from

    Returns:
        Tuple of (frame_data, frame_id, x, y, theta, camera_angle) or None if failed
        Returns 0 if graceful disconnect signal received
            frame_data: JPEG-encoded frame bytes
            frame_id: Frame sequence number
            x: Robot x position in mm (world coordinates)
            y: Robot y position in mm (world coordinates)
            theta: Robot orientation in radians
            camera_angle: Camera servo angle in radians
    """
    data = recv_message(sock)
    if data is None:
        return None

    # Check for graceful disconnect signal (single 0 byte)
    if len(data) == 1 and data[0] == 0:
        return 0

    if len(data) < 20:  # 4 bytes frame_id + 16 bytes (4 floats) + frame data
        return None

    frame_id = struct.unpack('!I', data[:4])[0]
    x, y, theta, camera_angle = struct.unpack('!ffff', data[4:20])
    frame_data = data[20:]
    return frame_data, frame_id, x, y, theta, camera_angle


def send_command(
        sock: socket.socket,
        msg_type: int,
        args: list[float],
        command_id: int = 0) -> bool:
    """
    Send a generic command message.

    Args:
        sock: Socket to send on
        msg_type: Message type identifier (0-255)
        args: List of float arguments
        command_id: Command ID for tracking completion (0 = no tracking)

    Returns:
        True if successful
    """
    try:
        # Validate message type
        if msg_type not in message_types.messageTypes:
            print(f"Unknown message type: {msg_type}")
            return False

        # Pack: 1-byte message type + 4-byte command_id + N 4-byte floats
        data = struct.pack('!B', msg_type) + \
            struct.pack('!I', command_id) + \
            struct.pack(f'!{len(args)}f', *args)
        return send_message(sock, data)
    except struct.error as e:
        print(f"Failed to pack command: {e}")
        return False


def recv_command(sock: socket.socket) -> Optional[Tuple[int, int, list[float]]]:
    """
    Receive a generic command message.

    Args:
        sock: Socket to receive from

    Returns:
        Tuple of (msg_type, command_id, args) where args is a list of floats, or None if failed
    """
    # Receive message with length header
    data = recv_message(sock)
    print(f'data is {data}')
    if data is None or len(data) < 5:  # At least 1 byte type + 4 byte command_id
        return None

    try:
        # Unpack message type (1 byte)
        msg_type: int = struct.unpack('!B', data[:1])[0]

        # Validate message type
        if msg_type not in message_types.messageTypes:
            print(f"Unknown message type: {msg_type}")
            return None

        # Unpack command_id (4 bytes)
        command_id: int = struct.unpack('!I', data[1:5])[0]

        # 1 byte type + 4 bytes command_id + N floats (4 bytes each) = length of data
        arg_count = (len(data) - 5) // 4

        # Unpack float arguments
        args = list(struct.unpack(f'!{arg_count}f', data[5:]))

        return msg_type, command_id, args
    except struct.error as e:
        print(f"Failed to unpack command: {e}")
        return None


def send_frame_info(
        sock: socket.socket,
        frame_info: FrameInfo,
        jpeg_quality: int = 80) -> bool:
    """
    Send a FrameInfo object with dual camera frames and robot state.

    Args:
        sock: Socket to send on
        frame_info: FrameInfo object containing frames and state
        jpeg_quality: JPEG compression quality (0-100)

    Returns:
        True if successful
    """
    try:
        # Encode frames to JPEG
        _, frame_top_encoded = cv2.imencode(
            '.jpg',
            frame_info.frame_top,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        _, frame_bottom_encoded = cv2.imencode(
            '.jpg',
            frame_info.frame_bottom,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )

        frame_top_bytes = frame_top_encoded.tobytes()
        frame_bottom_bytes = frame_bottom_encoded.tobytes()

        # Pack data:
        # - frame_id (4 bytes)
        # - 7 floats: x, y, theta, gripperHeight, gripperAngle, scooperAngle, distanceSensed (28 bytes)
        # - isMoving (1 byte boolean)
        # - lastCompletedCommandId (4 bytes)
        # - frame_top length (4 bytes)
        # - frame_top data (variable)
        # - frame_bottom length (4 bytes)
        # - frame_bottom data (variable)
        packet = struct.pack('!I', frame_info.frame_id)
        packet += struct.pack(
            '!fffffff',
            frame_info.x,
            frame_info.y,
            frame_info.theta,
            frame_info.gripperHeight,
            frame_info.gripperAngle,
            frame_info.scooperAngle,
            frame_info.distanceSensed
        )
        packet += struct.pack('!?', frame_info.isMoving)
        packet += struct.pack('!I', frame_info.lastCompletedCommandId)
        packet += struct.pack('!I', len(frame_top_bytes))
        packet += frame_top_bytes
        packet += struct.pack('!I', len(frame_bottom_bytes))
        packet += frame_bottom_bytes

        return send_message(sock, packet)
    except (cv2.error, struct.error) as e:
        print(f"Failed to send frame info: {e}")
        return False


def recv_frame_info(sock: socket.socket) -> Optional[FrameInfo | Literal[0]]:
    """
    Receive a FrameInfo object with dual camera frames and robot state.

    Args:
        sock: Socket to receive from

    Returns:
        FrameInfo object or None if failed
        Returns 0 if graceful disconnect signal received
    """
    data = recv_message(sock)
    if data is None:
        return None

    # Check for graceful disconnect signal (single 0 byte)
    if len(data) == 1 and data[0] == 0:
        print("Received graceful disconnect from Pi")
        return 0

    # Minimum size: 4 (frame_id) + 28 (7 floats) + 1 (isMoving) + 4 (lastCompletedCommandId) + 4 (top_len) + 4
    # (bottom_len) = 45 bytes
    if len(data) < 45:
        return None

    try:
        offset = 0

        # Unpack frame_id
        frame_id = struct.unpack('!I', data[offset:offset + 4])[0]
        offset += 4

        # Unpack 7 floats
        x, y, theta, gripperHeight, gripperAngle, scooperAngle, distanceSensed = \
            struct.unpack('!fffffff', data[offset:offset + 28])
        offset += 28

        # Unpack isMoving boolean
        is_moving = struct.unpack('!?', data[offset:offset + 1])[0]
        offset += 1

        # Unpack lastCompletedCommandId
        last_completed_command_id = struct.unpack('!I', data[offset:offset + 4])[0]
        offset += 4

        # Unpack frame_top
        frame_top_len = struct.unpack('!I', data[offset:offset + 4])[0]
        offset += 4
        frame_top_bytes = data[offset:offset + frame_top_len]
        offset += frame_top_len

        # Unpack frame_bottom
        frame_bottom_len = struct.unpack('!I', data[offset:offset + 4])[0]
        offset += 4
        frame_bottom_bytes = data[offset:offset + frame_bottom_len]

        # Decode JPEG frames
        frame_top = cv2.imdecode(
            np.frombuffer(frame_top_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        frame_bottom = cv2.imdecode(
            np.frombuffer(frame_bottom_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame_top is None or frame_bottom is None:
            print("Failed to decode JPEG frames")
            return None

        return FrameInfo(
            frame_top=frame_top,
            frame_bottom=frame_bottom,
            frame_id=frame_id,
            x=x,
            y=y,
            theta=theta,
            gripperHeight=gripperHeight,
            gripperAngle=gripperAngle,
            scooperAngle=scooperAngle,
            distanceSensed=distanceSensed,
            isMoving=is_moving,
            lastCompletedCommandId=last_completed_command_id
        )
    except (struct.error, cv2.error) as e:
        print(f"Failed to unpack frame info: {e}")
        return None

#!/usr/bin/env python3
"""
TCP Protocol - Length-prefixed message protocol for reliable communication
"""

import socket


class TCPProtocol:
    """Handles length-prefixed TCP message protocol"""

    HEADER_SIZE = 4  # 4 bytes for message length
    CHUNK_SIZE = 4096  # Size of chunks for receiving data

    @staticmethod
    def send_message(sock, message):
        """
        Send a length-prefixed message over a socket

        Args:
            sock: Socket object
            message: String message to send

        Raises:
            socket.error: If sending fails
        """
        # Encode message to bytes
        message_bytes = message.encode('utf-8')

        # Calculate length and create 4-byte prefix
        length = len(message_bytes)
        length_prefix = length.to_bytes(TCPProtocol.HEADER_SIZE, byteorder='big')

        # Send length prefix + message
        sock.sendall(length_prefix + message_bytes)

    @staticmethod
    def recv_message(sock):
        """
        Receive a length-prefixed message from a socket

        Args:
            sock: Socket object

        Returns:
            String message, or None if connection closed

        Raises:
            socket.error: If receiving fails
        """
        # First, receive the length prefix (4 bytes)
        length_data = b''
        while len(length_data) < TCPProtocol.HEADER_SIZE:
            chunk = sock.recv(TCPProtocol.HEADER_SIZE - len(length_data))
            if not chunk:
                return None  # Connection closed
            length_data += chunk

        # Decode the length
        message_length = int.from_bytes(length_data, byteorder='big')

        # Receive the actual message in chunks
        message_data = b''
        while len(message_data) < message_length:
            bytes_remaining = message_length - len(message_data)
            chunk_size = min(TCPProtocol.CHUNK_SIZE, bytes_remaining)
            chunk = sock.recv(chunk_size)
            if not chunk:
                return None  # Connection closed unexpectedly
            message_data += chunk

        # Decode and return the message
        return message_data.decode('utf-8')


class TCPConnection:
    """Wrapper for TCP socket connection with protocol support"""

    def __init__(self, sock=None):
        """
        Initialize TCP connection

        Args:
            sock: Existing socket object, or None to create new one
        """
        self.socket = sock if sock else socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

    def connect(self, host, port):
        """
        Connect to a TCP server

        Args:
            host: Server hostname or IP address
            port: Server port number

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.socket.connect((host, port))
            self.connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False

    def send(self, message):
        """
        Send a message using length-prefixed protocol

        Args:
            message: String message to send

        Returns:
            bool: True if send successful, False otherwise
        """
        if not self.connected:
            return False

        try:
            TCPProtocol.send_message(self.socket, message)
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            self.connected = False
            return False

    def receive(self):
        """
        Receive a message using length-prefixed protocol

        Returns:
            String message, or None if receive failed
        """
        if not self.connected:
            return None

        try:
            return TCPProtocol.recv_message(self.socket)
        except Exception as e:
            print(f"Error receiving message: {e}")
            self.connected = False
            return None

    def send_receive(self, message):
        """
        Send a message and receive response

        Args:
            message: String message to send

        Returns:
            String response, or None if failed
        """
        if self.send(message):
            return self.receive()
        return None

    def close(self):
        """Close the connection"""
        if self.socket:
            self.socket.close()
        self.connected = False

"""
Shared configuration for video streaming between Raspberry Pi and computer.
"""

# Network settings
PI_IP = "192.168.1.100"  # Change to your Pi's IP address
COMPUTER_IP = "10.42.0.179"  # Change to your computer's IP address

# Ports
VIDEO_PORT = 5000  # Port for video streaming
COMMAND_PORT = 5001  # Port for command data

# Video settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80  # 0-100, higher = better quality but more bandwidth
DEFAULT_MAX_FPS = 30.0  # Default maximum frames per second for streaming

# Protocol settings
HEADER_SIZE = 8  # bytes for message length header
BUFFER_SIZE = 65536  # receive buffer size

# Timeouts (seconds)
SOCKET_TIMEOUT = 180.0  # 3 mins, longer than match time
RECONNECT_DELAY = 3.0  # Delay between reconnection attempts

FT_TO_MM = 304.8

# measurements are in mm
CENTER_BORDER_X = 8 * FT_TO_MM
BACK_BORDER_X = 0
LEFT_BORDER_Y = 4 * FT_TO_MM
RIGHT_BORDER_Y = - 4 * FT_TO_MM

WHEEL_D = 101.6
BASE_D = 237.236

CAN_DIAMETER = 76.2
CAN_HEIGHT = 122.5  # Standard can height in mm

SCOOPER_LENGTH = 254
CLAW_OFFSET = 120


# --------------------- NETWORKING ---------------------

# Network settings
PI_IP = "192.168.1.100"  # Change to your Pi's IP address
COMPUTER_IP = "10.42.0.61"  # Change to your computer's IP address
# COMPUTER_IP = "10.42.0.210"  # Change to your computer's IP address

# Ports
VIDEO_PORT = 9000  # Port for video streaming
COMMAND_PORT = 9001  # Port for command data

# Video settings
# networking likes multiples of 32 for some reason,
# when i put 854, it sent 864 instead
FRAME_WIDTH = 864
FRAME_HEIGHT = 480
JPEG_QUALITY = 80  # 0-100, higher = better quality but more bandwidth
FPS = 30.0  # Default maximum frames per second for streaming

# Protocol settings
HEADER_SIZE = 8  # bytes for message length header
BUFFER_SIZE = 65536  # receive buffer size

# Timeouts (seconds)
SOCKET_TIMEOUT = 180.0  # 3 mins, longer than match time
RECONNECT_DELAY = 3.0  # Delay between reconnection attempts

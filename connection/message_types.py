# Network Message Types
CLOSE = 0  # Close connection (no arguments)
# Movement command: [left_coef, right_coef, distance]
ADD_MOVEMENT = 1  # One movement command
OVERRIDE_MOVEMENTS = 2  # List of movement commands
SEND_WORLD_XY = 3  # World coordinate navigation: [world_x, world_y]
GRIP_CAN = 4  # Grip can and lift: [height_mm]
RELEASE_CAN = 5  # Release can grip: [height_mm]
SEND_GRIPPER_HEIGHT = 6  # Set gripper height: [height_mm]

messageTypes = [
    CLOSE,
    ADD_MOVEMENT,
    OVERRIDE_MOVEMENTS,
    SEND_WORLD_XY,
    GRIP_CAN,
    RELEASE_CAN,
    SEND_GRIPPER_HEIGHT
]

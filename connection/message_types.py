# Network Message Types
CLOSE = 0  # Close connection (no arguments)
# Movement command: [left_coef, right_coef, distance]
ADD_MOVEMENT = 1  # One movement command
OVERRIDE_MOVEMENTS = 2  # List of movement commands
SEND_WORLD_XY = 3  # World coordinate navigation: [world_x, world_y]

messageTypes = [
    CLOSE,
    ADD_MOVEMENT,
    OVERRIDE_MOVEMENTS,
    SEND_WORLD_XY
]

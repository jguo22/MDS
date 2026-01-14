# Network Message Types
CLOSE = 0  # Close connection (no arguments)
# Movement command: [left_coef, right_coef, distance]
ADD_MOVEMENT = 1  # One movement command
OVERRIDE_MOVEMENTS = 2  # List of movement commands

messageTypes = [
    CLOSE,
    ADD_MOVEMENT,
    OVERRIDE_MOVEMENTS
]

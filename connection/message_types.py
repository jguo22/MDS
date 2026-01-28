# Network Message Types
CLOSE = 0  # Close connection (no arguments)
PING = 1  # Latency test ping: [timestamp]
# List of movement commands: [left_coef, right_coef, distance, ...]
OVERRIDE_MOVEMENTS = 2
# Waypoint navigation: [start_x, start_y, wp1_x, wp1_y, ...]
OVERRIDE_WAYPOINTS = 3
PICKUP_CAN = 4  # Pick up can (no arguments)
RELEASE_CAN = 5  # Release can (no arguments)
# Early game strategy: [golden_x, golden_y, left_x, left_y, right_x, right_y]
EARLY_GAME = 6
OVERRIDE_RELATIVE_XY = 7  # Relative movement: [x, y] in mm
OVERRIDE_WORLD_XY = 8  # World coordinate navigation: [world_x, world_y] in mm

messageTypes = [
    CLOSE,
    PING,
    OVERRIDE_MOVEMENTS,
    OVERRIDE_WAYPOINTS,
    PICKUP_CAN,
    RELEASE_CAN,
    EARLY_GAME,
    OVERRIDE_RELATIVE_XY,
    OVERRIDE_WORLD_XY
]

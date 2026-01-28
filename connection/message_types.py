# Network Message Types
CLOSE = 0  # Close connection (no arguments)
PING = 1  # Latency test ping: [timestamp]
OVERRIDE_MOVEMENTS = 2
OVERRIDE_WAYPOINTS = 3
PICKUP_CAN = 4
RELEASE_CAN = 5
EARLY_GAME = 6

messageTypes = [
    CLOSE,
    PING,
    OVERRIDE_MOVEMENTS,
    OVERRIDE_WAYPOINTS,
    PICKUP_CAN,
    RELEASE_CAN,
    EARLY_GAME
]

"""
Simple command ID tracking with module-level variables.
Thread-safe tracking for command completion state.
"""

import threading

# Module-level variables for command tracking
_lock = threading.Lock()
_last_completed_command_id = 0


def mark_complete(command_id: int) -> None:
    """
    Mark a command as complete.

    Args:
        command_id: The ID of the completed command
    """
    global _last_completed_command_id
    with _lock:
        if command_id > _last_completed_command_id:
            _last_completed_command_id = command_id


def get_last_completed_command_id() -> int:
    """
    Get the ID of the last completed command.

    Returns:
        Last completed command ID (0 if none)
    """
    with _lock:
        return _last_completed_command_id

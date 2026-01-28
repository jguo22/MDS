"""
Abstract interface for robot command execution.

Defines the common interface for sending commands to the robot,
whether over network (ComputerReceiver) or directly (DirectRobotCommander).
"""

from abc import ABC, abstractmethod
from typing import Tuple


class IRobotCommander(ABC):
    """Abstract base class for robot command execution."""

    @abstractmethod
    def early_game(
            self,
            golden: Tuple[float,
                          float],
            left: Tuple[float,
                        float],
            right: Tuple[float,
                         float]) -> bool:
        """
        Send early game strategy command with can locations.

        Args:
            golden: (x, y) coordinates of golden can in mm
            left: (x, y) coordinates of left can in mm
            right: (x, y) coordinates of right can in mm

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def override_movement(self, movement_args: list[float]) -> bool:
        """
        Override current path with new movement commands.

        Args:
            movement_args: List of movement commands in groups of 3 floats:
                [left_coef, right_coef, distance, ...]
                - left_coef: Left motor coefficient (-1.0 to 1.0)
                - right_coef: Right motor coefficient (-1.0 to 1.0)
                - distance: Distance to move in encoder ticks

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def override_waypoints(self, movement_args: list[float]) -> bool:
        """
        Override current path with waypoint navigation.

        Args:
            movement_args: List of coordinates [start_x, start_y, wp1_x, wp1_y, ...]
                - First two values are the starting point
                - Remaining pairs are waypoints to navigate through

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def pickup_can(self) -> bool:
        """
        Pick up a can with the gripper.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def release_can(self) -> bool:
        """
        Release the can from the gripper.

        Returns:
            True if successful
        """
        pass

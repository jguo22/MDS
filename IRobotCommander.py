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
    def override_relative_xy(self, x: float, y: float) -> bool:
        """
        Override current path with relative movement in ROS coordinates.

        Args:
            x: Forward distance in mm (positive = forward, negative = backward)
            y: Lateral distance in mm (positive = left, negative = right)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def override_world_xy(self, world_x: float, world_y: float) -> bool:
        """
        Override current path to navigate to world coordinates.

        Args:
            world_x: Target x position in world frame (mm)
            world_y: Target y position in world frame (mm)

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

    @abstractmethod
    def approach_can_with_ds(self) -> bool:
        """
        Approach can using distance sensor feedback.

        Uses distance sensor to approach can in real-time, stopping when
        within 100mm or returning False if no can detected (> 800mm).

        Returns:
            True if successfully approached can, False if no can detected
        """
        pass

    @abstractmethod
    def stack(
            self,
            temp_pos: Tuple[float, float],
            stack_pos: Tuple[float, float],
            stacked_cans: int) -> bool:
        """
        Stack can at temporary position then stack with existing cans.

        Args:
            temp_pos: (x, y) temporary position to place can in mm
            stack_pos: (x, y) position of stack in mm
            stacked_cans: Number of cans already stacked

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def waitFinishedMoving(self) -> bool:
        """
        Wait for current movement to complete.

        Blocks until the robot's navigation system reports no movement.

        Returns:
            True if successful
        """
        pass

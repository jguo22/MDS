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
    def send_early_game(
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
    def send_grip_can(self, height_mm: float) -> bool:
        """
        Send command to grip can and lift to specified height.

        Args:
            height_mm: Height to lift gripper to in mm

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def send_release_can(self, height_mm: float) -> bool:
        """
        Send command to lower gripper and release can.

        Args:
            height_mm: Height to position gripper at before releasing in mm

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def send_gripper_height(self, height_mm: float) -> bool:
        """
        Send command to set gripper height.

        Args:
            height_mm: Height to set gripper to in mm

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def send_world_xy(self, world_x: float, world_y: float) -> bool:
        """
        Send world coordinate navigation command.

        Args:
            world_x: Target x position in world frame (mm)
            world_y: Target y position in world frame (mm)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def add_movement(
            self,
            left_coef: float,
            right_coef: float,
            distance: float) -> bool:
        """
        Add a single movement command to the queue.

        Args:
            left_coef: Left motor coefficient (-1.0 to 1.0)
            right_coef: Right motor coefficient (-1.0 to 1.0)
            distance: Distance to move in encoder ticks

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def send_xy(self, x: float, y: float) -> bool:
        """
        Send relative movement in ROS coordinates.

        Args:
            x: Forward distance in mm (positive = forward)
            y: Lateral distance in mm (positive = left)

        Returns:
            True if successful
        """
        pass

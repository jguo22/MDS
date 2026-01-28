"""
Direct robot commander that executes commands locally without network communication.

This class implements IRobotCommander by directly calling the navigation
and hardware control functions, suitable for Pi-side execution.
"""

import math
from typing import Tuple
from IRobotCommander import IRobotCommander
from nav import Nav, NavMove
from distanceSensorWrapper import DistanceSensorWrapper
from RavenWrapper import RAVEN_WRAPPER
from earlyGame import EarlyGame
import navHelpers


class DirectRobotCommander(IRobotCommander):
    """
    Direct implementation of IRobotCommander for local execution.

    Executes robot commands directly by calling nav, RAVEN_WRAPPER,
    and other hardware interfaces without network overhead.
    """

    def __init__(self, nav: Nav, distance_sensor: DistanceSensorWrapper):
        """
        Initialize direct robot commander.

        Args:
            nav: Navigation controller instance
            distance_sensor: Distance sensor wrapper instance
        """
        self.nav = nav
        self.distance_sensor = distance_sensor

    def send_early_game(
            self,
            golden: Tuple[float,
                          float],
            left: Tuple[float,
                        float],
            right: Tuple[float,
                         float]) -> bool:
        """
        Execute early game strategy with can locations.

        Args:
            golden: (x, y) coordinates of golden can in mm
            left: (x, y) coordinates of left can in mm
            right: (x, y) coordinates of right can in mm

        Returns:
            True if successful
        """
        try:
            early_game = EarlyGame(
                self.nav,
                self.distance_sensor,
                golden,
                left,
                right)
            early_game.performEarlyGame()
            return True
        except Exception as e:
            print(f"Error in send_early_game: {e}")
            return False

    def override_movement(self, movement_args: list[float]) -> bool:
        """
        Override current path with new movement commands.

        Args:
            movement_args: List of movement commands in groups of 3 floats:
                [left_coef, right_coef, distance, ...]

        Returns:
            True if successful
        """
        try:
            assert len(movement_args) % 3 == 0
            moves = []
            for i in range(len(movement_args) // 3):
                moves.append(
                    NavMove(
                        movement_args[3 * i],
                        movement_args[3 * i + 1],
                        movement_args[3 * i + 2],
                        False))
            self.nav.overridePaths(moves)
            return True
        except Exception as e:
            print(f"Error in override_movement: {e}")
            return False

    def send_grip_can(self, height_mm: float) -> bool:
        """
        Grip can and lift to specified height.

        Args:
            height_mm: Height to lift gripper to in mm

        Returns:
            True if successful
        """
        try:
            RAVEN_WRAPPER.close_gripper()
            RAVEN_WRAPPER.raise_elevator()
            return True
        except Exception as e:
            print(f"Error in send_grip_can: {e}")
            return False

    def send_release_can(self, height_mm: float) -> bool:
        """
        Lower gripper and release can.

        Args:
            height_mm: Height to position gripper at before releasing in mm

        Returns:
            True if successful
        """
        try:
            RAVEN_WRAPPER.lower_elevator()
            RAVEN_WRAPPER.open_gripper()
            return True
        except Exception as e:
            print(f"Error in send_release_can: {e}")
            return False

    def send_gripper_height(self, height_mm: float) -> bool:
        """
        Set gripper height.

        Args:
            height_mm: Height to set gripper to in mm

        Returns:
            True if successful
        """
        try:
            print(f"Setting gripper height: {height_mm}mm")
            RAVEN_WRAPPER.raise_elevator()
            return True
        except Exception as e:
            print(f"Error in send_gripper_height: {e}")
            return False

    def send_world_xy(self, world_x: float, world_y: float) -> bool:
        """
        Navigate to world coordinates.

        Args:
            world_x: Target x position in world frame (mm)
            world_y: Target y position in world frame (mm)

        Returns:
            True if successful
        """
        try:
            print(f"Navigating to world coordinates: x={world_x}, y={world_y}")
            self.nav.override_paths_world_xy(world_x, world_y)
            return True
        except Exception as e:
            print(f"Error in send_world_xy: {e}")
            return False

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
        try:
            print(
                f"Adding movement: left={left_coef}, right={right_coef}, dist={distance}")
            self.nav.addPath(NavMove(left_coef, right_coef, distance, False))
            return True
        except Exception as e:
            print(f"Error in add_movement: {e}")
            return False

    def send_xy(self, x: float, y: float) -> bool:
        """
        Send relative movement in ROS coordinates.

        Args:
            x: Forward distance in mm (positive = forward)
            y: Lateral distance in mm (positive = left)

        Returns:
            True if successful
        """
        try:
            distance = math.sqrt(x * x + y * y)
            theta = math.atan2(y, x)

            rotate = list(navHelpers.get_rotate(theta))
            forward = list(navHelpers.get_forward_mm(distance))

            print(
                f'Sending movement: x={x} y={y} theta={theta} distance={distance}')

            return self.override_movement(rotate + forward)
        except Exception as e:
            print(f"Error in send_xy: {e}")
            return False

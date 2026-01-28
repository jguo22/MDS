"""
Direct robot commander that executes commands locally without network communication.

This class implements IRobotCommander by directly calling the navigation
and hardware control functions, suitable for Pi-side execution.
"""

from typing import Tuple
import math

from spatialmath import SE2
from IMUWrapper import IMUWrapper
from IRobotCommander import IRobotCommander
from config import BASE_D
from nav import Nav, NavMove
from distanceSensorWrapper import DistanceSensorWrapper
from RavenWrapper import RAVEN_WRAPPER
from earlyGame import EarlyGame
from navHelpers import get_forward_mm, get_rotate
from vision.relativeCoordinates import get_movement_plan


class DirectRobotCommander(IRobotCommander):
    """
    Direct implementation of IRobotCommander for local execution.

    Executes robot commands directly by calling nav, RAVEN_WRAPPER,
    and other hardware interfaces without network overhead.
    """

    def __init__(
            self,
            nav: Nav,
            distance_sensor: DistanceSensorWrapper,
            imu: IMUWrapper):
        """
        Initialize direct robot commander.

        Args:
            nav: Navigation controller instance
            distance_sensor: Distance sensor wrapper instance
        """
        self.nav = nav
        self.distance_sensor = distance_sensor
        self.imu = imu
        self.previous_location = (0, 0)

    def early_game(
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

    def override_waypoints(self, movement_args: list[float]) -> bool:
        """
        Receive a list of x, y coordinates and navigate through waypoints.

        Args:
            movement_args: List of coordinates [start_x, start_y, wp1_x, wp1_y, wp2_x, wp2_y, ...]
                - First two values are the starting point (current robot position)
                - Remaining pairs are waypoints to navigate to in sequence

        Returns:
            True if successful
        """
        try:
            assert len(
                movement_args) % 2 == 0, "Movement args must be pairs of x, y coordinates"

            # Parse starting point and waypoints
            start = movement_args[:2]
            waypoints = []
            for i in range(2, len(movement_args), 2):
                waypoints.append((movement_args[i], movement_args[i + 1]))

            x, y = RAVEN_WRAPPER.get_odometry()
            theta = self.imu.get_heading()
            robot_pose = SE2(x, y, theta)

            # remove the first waypoint if we already passed it
            # within the ~3 frames delay
            if is_near_segment(start, [x, y], waypoints[0], BASE_D):
                waypoints.pop(0)

            plan = get_movement_plan(waypoints, robot_pose)

            movement_args = []
            for move in plan:
                dist, theta = move
                if theta > 0.01:
                    movement_args.extend(get_rotate(theta))
                if dist > 5:
                    movement_args.extend(get_forward_mm(dist))
            return self.override_movement(movement_args)

        except Exception as e:
            print(f"Error in override_waypoints: {e}")
            return False

    def override_relative_xy(self, x: float, y: float) -> bool:
        """
        Override current path with relative movement in ROS coordinates.

        Args:
            x: Forward distance in mm (positive = forward, negative = backward)
            y: Lateral distance in mm (positive = left, negative = right)

        Returns:
            True if successful
        """
        try:
            distance = math.sqrt(x * x + y * y)

            # ROS coordinates: x is forward, y is left
            # atan2(y, x) gives angle from forward axis (x) to target
            theta = math.atan2(y, x)

            rotate_move = get_rotate(theta)
            forward_move = get_forward_mm(distance)

            print(
                f'Relative xy movement: x={x} y={y} theta={theta:.3f} distance={distance:.1f}')

            movement_args = list(rotate_move) + list(forward_move)
            return self.override_movement(movement_args)

        except Exception as e:
            print(f"Error in override_relative_xy: {e}")
            return False

    def override_world_xy(self, world_x: float, world_y: float) -> bool:
        """
        Override current path to navigate to world coordinates.

        Args:
            world_x: Target x position in world frame (mm)
            world_y: Target y position in world frame (mm)

        Returns:
            True if successful
        """
        try:
            # Get current robot position from nav odometry
            current_x, current_y = RAVEN_WRAPPER.get_odometry()
            current_heading = self.imu.get_heading()

            # Calculate relative position to target
            dx = world_x - current_x
            dy = world_y - current_y
            distance = math.sqrt(dx * dx + dy * dy)

            # Calculate target heading in world frame
            target_heading = math.atan2(dy, dx)

            # Calculate rotation needed from current heading
            rotation_angle = target_heading - current_heading

            rotate_move = get_rotate(rotation_angle)
            forward_move = get_forward_mm(distance)

            print(f'World xy movement: target=({world_x}, {world_y}) current=({current_x:.1f}, {current_y:.1f}) '
                  f'rotate={rotation_angle:.3f}rad distance={distance:.1f}mm')

            movement_args = list(rotate_move) + list(forward_move)
            return self.override_movement(movement_args)

        except Exception as e:
            print(f"Error in override_world_xy: {e}")
            return False

    def pickup_can(self) -> bool:
        RAVEN_WRAPPER.lower_elevator()
        RAVEN_WRAPPER.close_gripper()
        RAVEN_WRAPPER.raise_elevator()
        return True

    def release_can(self) -> bool:
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


def is_near_segment(A, B, P, r):
    # Vector math: A=start, B=end, P=test point, r=radius
    dx, dy = B[0] - A[0], B[1] - A[1]
    mag_sq = dx**2 + dy**2

    if mag_sq == 0:
        return math.dist(A, P) <= r

    # Projection parameter t clamped to [0, 1]
    t = max(0, min(1, ((P[0] - A[0]) * dx + (P[1] - A[1]) * dy) / mag_sq))

    # Closest point on segment
    closest = (A[0] + t * dx, A[1] + t * dy)

    return math.dist(P, closest) <= r

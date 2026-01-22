"""
Coordinate transformation utilities using spatialmath SE(2) representation.

This module provides functions for converting between world coordinates and
robot-relative coordinates using SE(2) (Special Euclidean group in 2D)
transformations. SE(2) represents rigid body transformations in 2D space,
combining both translation (x, y) and rotation (theta).

Uses the spatialmath-python library for robust 2D rigid body transformations.
Install: pip install spatialmath-python

Coordinate Systems:
------------------
World coordinate system:
- Origin: fixed reference point (e.g., starting position)
- x-axis: forward direction (positive = forward)
- y-axis: left direction (positive = left)
- theta=0: facing forward (positive x direction)
- Units: millimeters (mm) for position, radians for orientation

Relative (robot-centric) coordinate system:
- Origin: robot's current position
- x-axis: robot's forward direction (positive = ahead of robot)
- y-axis: robot's left direction (positive = left of robot)
- theta=0: aligned with robot's heading
- Coordinates are relative to robot's current pose (position + orientation)
- Units: millimeters (mm) for position, radians for orientation

Key Functions:
--------------
- world_to_relative(): Convert world point to robot-relative coordinates
- relative_to_world(): Convert robot-relative point to world coordinates
- transform_pose_world_to_relative(): Transform full pose (position + orientation)
- transform_pose_relative_to_world(): Transform full pose back to world frame
- angle_between_points(): Calculate angle of line between two points

All functions use SE(2) transformations for mathematically robust coordinate
conversions that properly handle both translation and rotation.
"""

import numpy as np
from spatialmath import SE2
from typing import List, Optional, Tuple, Union
import math


def _point_to_array(
        point: Union[Tuple[float, float], np.ndarray, SE2]) -> np.ndarray:
    """
    Extract (x, y) coordinates from various point representations.

    This is a helper function that normalizes different point input types
    into a consistent numpy array format for use in coordinate transformations.

    Args:
        point: Point representation. Can be:
            - Tuple (x, y) in mm
            - numpy array [x, y] in mm
            - SE2 pose (uses translation component, ignores orientation)

    Returns:
        numpy array [x, y] as float64 with shape (2,)

    Note:
        For SE2 inputs, only the translation component (x, y position) is extracted.
        The orientation component is ignored.
    """
    if isinstance(point, SE2):
        return point.t
    elif isinstance(point, tuple):
        return np.array(point, dtype=float)
    else:
        return np.asarray(point, dtype=float)


def world_to_relative(
    world_point: Union[Tuple[float, float], np.ndarray, SE2],
    robot_pose: SE2
) -> Tuple[float, float]:
    """
    Convert world coordinates to robot-relative coordinates using SE(2).

    Transforms a point from the world coordinate frame to the robot's local
    coordinate frame. The robot frame has its origin at the robot's position
    and is oriented along the robot's heading direction.

    Mathematical operation: rel_point = robot_pose.inv() * world_point

    Args:
        world_point: Target point in world frame. Can be:
            - Tuple (x, y) in mm
            - numpy array [x, y] in mm
            - SE2 pose (uses translation component)
        robot_pose: Robot pose in world frame as SE2 object
            - Translation: (x, y) position in world frame (mm)
            - Rotation: orientation in world frame (radians)

    Returns:
        Tuple of (rel_x, rel_y) in robot-relative frame (mm)
        - rel_x: distance forward from robot (positive = ahead)
        - rel_y: distance left from robot (positive = left)

    Example:
        >>> robot = SE2(0, 0, 0)  # Robot at origin facing +x
        >>> x, y = world_to_relative((1000, 500), robot)
        >>> x, y
        (1000.0, 500.0)

        >>> robot = SE2(1000, 0, 0)  # Robot at (1000, 0) facing +x
        >>> x, y = world_to_relative((0, 0), robot)
        >>> x, y
        (-1000.0, 0.0)
    """
    # Convert point to numpy array for transformation
    world_point = _point_to_array(world_point)

    # Transform point from world frame to robot frame
    rel_point: np.ndarray = robot_pose.inv() * world_point

    return rel_point[0].item(), rel_point[1].item()


def relative_to_world(
    rel_point: Union[Tuple[float, float], np.ndarray],
    robot_pose: SE2
) -> Tuple[float, float]:
    """
    Convert robot-relative coordinates to world coordinates using SE(2).

    Transforms a point from the robot's local coordinate frame to the world
    coordinate frame. This is the inverse operation of world_to_relative().

    Mathematical operation: world_point = robot_pose * rel_point

    Args:
        rel_point: Point in robot-relative frame. Can be:
            - Tuple (x, y) in mm
            - numpy array [x, y] in mm
            Where x is forward from robot, y is left from robot
        robot_pose: Robot pose in world frame as SE2 object
            - Translation: (x, y) position in world frame (mm)
            - Rotation: orientation in world frame (radians)

    Returns:
        Tuple of (world_x, world_y) in world frame (mm)
        - world_x: x position in world frame (forward direction)
        - world_y: y position in world frame (left direction)

    Example:
        >>> robot = SE2(0, 0, 0)  # Robot at origin facing +x
        >>> x, y = relative_to_world((1000, 0), robot)
        >>> x, y
        (1000.0, 0.0)

        >>> robot = SE2(1000, 0, 0)  # Robot at (1000, 0) facing +x
        >>> x, y = relative_to_world((0, 500), robot)
        >>> x, y
        (1000.0, 500.0)
    """
    # Convert point to numpy array for transformation
    rel_point = _point_to_array(rel_point)

    # Transform point from robot frame to world frame
    world_point: np.ndarray = robot_pose * rel_point

    return world_point[0].item(), world_point[1].item()


def transform_pose_world_to_relative(
    world_pose: SE2,
    robot_pose: SE2
) -> SE2:
    """
    Convert a pose (position + orientation) from world frame to robot-relative frame.

    Transforms both the position and orientation of a target pose from the world
    coordinate frame to the robot's local coordinate frame. This is useful when
    you need to know not just where a target is relative to the robot, but also
    what direction it is facing relative to the robot's heading.

    Mathematical operation: robot_T_target = world_T_robot.inv() * world_T_target

    Args:
        world_pose: Target pose in world frame as SE2 object
            - Translation: (x, y) position in world frame (mm)
            - Rotation: orientation in world frame (radians)
        robot_pose: Robot pose in world frame as SE2 object
            - Translation: (x, y) position in world frame (mm)
            - Rotation: orientation in world frame (radians)

    Returns:
        SE2: Pose in robot-relative frame
        - Translation: (rel_x, rel_y) relative to robot (mm)
        - Rotation: orientation relative to robot's heading (radians)

    Example:
        >>> robot = SE2(0, 0, 0)  # Robot at origin facing +x
        >>> target = SE2(1000, 0, 0)  # Target at (1000, 0) facing +x
        >>> rel_pose = transform_pose_world_to_relative(target, robot)
        >>> rel_pose.t, rel_pose.theta()
        (array([1000.,    0.]), 0.0)

        >>> robot = SE2(500, 0, 0)  # Robot at (500, 0) facing +x
        >>> target = SE2(1000, 0, 0)  # Target at (1000, 0) facing +x
        >>> rel_pose = transform_pose_world_to_relative(target, robot)
        >>> rel_pose.t, rel_pose.theta()
        (array([500.,   0.]), 0.0)
    """
    # Compute relative transformation: robot_T_target = world_T_robot.inv() *
    # world_T_target
    robot_T_target: SE2 = robot_pose.inv() * world_pose

    return robot_T_target


def transform_pose_relative_to_world(
    rel_pose: SE2,
    robot_pose: SE2
) -> SE2:
    """
    Convert a pose (position + orientation) from robot-relative frame to world frame.

    Transforms both the position and orientation of a target pose from the robot's
    local coordinate frame to the world coordinate frame. This is the inverse
    operation of transform_pose_world_to_relative().

    Mathematical operation: world_T_target = world_T_robot * robot_T_target

    Args:
        rel_pose: Pose in robot-relative frame as SE2 object
            - Translation: (rel_x, rel_y) relative to robot (mm)
            - Rotation: orientation relative to robot's heading (radians)
        robot_pose: Robot pose in world frame as SE2 object
            - Translation: (x, y) position in world frame (mm)
            - Rotation: orientation in world frame (radians)

    Returns:
        SE2: Pose in world frame
        - Translation: (x, y) position in world frame (mm)
        - Rotation: orientation in world frame (radians)

    Example:
        >>> robot = SE2(0, 0, 0)  # Robot at origin facing +x
        >>> rel_pose = SE2(1000, 0, 0)  # Point 1000mm in front of robot
        >>> world_pose = transform_pose_relative_to_world(rel_pose, robot)
        >>> world_pose.t, world_pose.theta()
        (array([1000.,    0.]), 0.0)

        >>> robot = SE2(500, 0, 0)  # Robot at (500, 0) facing +x
        >>> rel_pose = SE2(0, 0, 0)  # Robot's own position
        >>> world_pose = transform_pose_relative_to_world(rel_pose, robot)
        >>> world_pose.t, world_pose.theta()
        (array([500.,   0.]), 0.0)
    """
    # Compute world transformation: world_T_target = world_T_robot *
    # robot_T_target
    world_T_target: SE2 = robot_pose * rel_pose

    return world_T_target


def angle_between_points(
    point1: Union[Tuple[float, float], np.ndarray, SE2],
    point2: Union[Tuple[float, float], np.ndarray, SE2],
) -> float:
    """
    Calculate the angle of the line from point1 to point2 relative to the x-axis.

    Computes the angle of the vector pointing from point1 to point2, measured
    from the positive x-axis. Uses atan2(dy, dx) for robust angle calculation
    that handles all quadrants correctly.

    The angle is measured from the positive x-axis with counter-clockwise being
    positive, following the standard mathematical convention.

    Args:
        point1: First point (origin of the vector). Can be:
            - Tuple (x, y) in mm
            - numpy array [x, y] in mm
            - SE2 pose (uses translation component, ignores orientation)
        point2: Second point (destination of the vector). Can be:
            - Tuple (x, y) in mm
            - numpy array [x, y] in mm
            - SE2 pose (uses translation component, ignores orientation)

    Returns:
        float: Angle in radians in range [-π, π]
        - Positive values: counter-clockwise rotation from x-axis
        - Negative values: clockwise rotation from x-axis
        - 0: pointing along positive x-axis
        - π/2: pointing along positive y-axis
        - -π/2: pointing along negative y-axis

    Examples:
        # Angle from (0,0) to (1,1) is π/4 radians (45 degrees)
        >>> angle_between_points((0, 0), (1, 1))
        0.7853981633974483  # π/4

        # Angle from (1,1) to (0,0) is -3π/4 radians (opposite direction)
        >>> angle_between_points((1, 1), (0, 0))
        -2.356194490192345  # -3π/4

        # Angle from (0,0) to (0,1) is π/2 radians (90 degrees, straight up)
        >>> angle_between_points((0, 0), (0, 1))
        1.5707963267948966  # π/2

        # Using SE2 poses
        >>> p1 = SE2(0, 0, 0)
        >>> p2 = SE2(1, 1, 0)
        >>> angle_between_points(p1, p2)
        0.7853981633974483  # π/4
    """
    point1_arr = _point_to_array(point1)
    point2_arr = _point_to_array(point2)

    dx = point2_arr[0] - point1_arr[0]
    dy = point2_arr[1] - point1_arr[1]
    angle = math.atan2(dy, dx)

    return angle


def plan_path_poses(
    points: List[Union[Tuple[float, float], np.ndarray]],
    start_pose: SE2
) -> List[SE2]:
    """
    Generate relative movement poses to navigate through world points sequentially.

    Converts world points into robot-relative movement commands (SE2 poses).
    Each pose represents the relative distance and angle to move from the current
    position to the next point.

    Args:
        points: World points to visit [(x, y), ...] in mm
        start_pose: Starting pose as SE2 (world position and orientation)

    Returns:
        List[SE2]: Relative poses for each movement step

    Example:
        start = SE2(0, 0, 0)
        points = [(1000, 0), (1000, 1000)]
        path = plan_path_poses(points, start)
        # Returns relative movements to reach each point
    """

    if not points:
        return []

    # Convert all points to numpy arrays for consistency
    point_array = [_point_to_array(p) for p in points]

    # Initialize the path with the starting pose
    path = []

    # Process each point
    prev_pose = start_pose
    for curr_point in point_array:
        # get the relative pose for movement
        rel_x, rel_y = world_to_relative(curr_point, prev_pose)
        theta = math.atan2(rel_y, rel_x)
        path.append(SE2(rel_x, rel_y, theta))

        # update prev_pose in world coordinates
        x, y = curr_point
        world_theta = prev_pose.theta() + theta
        prev_pose = SE2(x, y, world_theta)

    return path

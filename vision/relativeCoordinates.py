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
from typing import List, Tuple, Union
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
    # Convert point to numpy array for transformation
    world_point = _point_to_array(world_point)

    # Transform point from world frame to robot frame
    rel_point: np.ndarray = robot_pose.inv() * world_point

    return rel_point[0].item(), rel_point[1].item()


def relative_to_world(
    rel_point: Union[Tuple[float, float], np.ndarray],
    robot_pose: SE2
) -> Tuple[float, float]:
    # Convert point to numpy array for transformation
    rel_point = _point_to_array(rel_point)

    # Transform point from robot frame to world frame
    world_point: np.ndarray = robot_pose * rel_point

    return world_point[0].item(), world_point[1].item()


def transform_pose_world_to_relative(
    world_pose: SE2,
    robot_pose: SE2
) -> SE2:
    # Compute relative transformation: robot_T_target = world_T_robot.inv() *
    # world_T_target
    robot_T_target: SE2 = robot_pose.inv() * world_pose

    return robot_T_target


def transform_pose_relative_to_world(
    rel_pose: SE2,
    robot_pose: SE2
) -> SE2:
    # Compute world transformation: world_T_target = world_T_robot *
    # robot_T_target
    world_T_target: SE2 = robot_pose * rel_pose

    return world_T_target


def angle_of_segment(
    point1: Union[Tuple[float, float], np.ndarray, SE2],
    point2: Union[Tuple[float, float], np.ndarray, SE2],
) -> float:
    point1_arr = _point_to_array(point1)
    point2_arr = _point_to_array(point2)

    dx = point2_arr[0] - point1_arr[0]
    dy = point2_arr[1] - point1_arr[1]
    angle = math.atan2(dy, dx)

    return angle


def get_movement_plan(
    points: List[Tuple[float, float]],
    start_pose: SE2
) -> List[Tuple[float, float]]:
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
        relX, relY = world_to_relative(curr_point, prev_pose)
        dist = math.sqrt(relX * relX + relY * relY)
        theta = math.atan2(relY, relX)
        path.append((dist, theta))

        # update prev_pose in world coordinates
        x, y = curr_point
        world_theta = prev_pose.theta() + theta
        prev_pose = SE2(x, y, world_theta)

    return path

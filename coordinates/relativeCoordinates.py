"""
Coordinate transformation utilities using spatialmath SE(2) representation.

Uses the spatialmath-python library for robust 2D rigid body transformations.
Install: pip install spatialmath-python

World coordinate system:
- x-axis: forward
- y-axis: left
- theta=0: facing forward (positive x direction)

Relative (robot-centric) coordinate system:
- Origin at robot's current position
- x-axis: robot's forward direction
- y-axis: robot's left direction
- Coordinates relative to robot's current pose
"""

import numpy as np
from spatialmath import SE2
from typing import Tuple


def world_to_relative(
    world_x: float,
    world_y: float,
    robot_x: float,
    robot_y: float,
    robot_theta: float
) -> Tuple[float, float]:
    """
    Convert world coordinates to robot-relative coordinates using SE(2).

    Args:
        world_x: Target x position in world frame (mm)
        world_y: Target y position in world frame (mm)
        robot_x: Robot x position in world frame (mm)
        robot_y: Robot y position in world frame (mm)
        robot_theta: Robot orientation in world frame (radians, 0 = facing forward/+x)

    Returns:
        Tuple of (rel_x, rel_y) in robot-relative frame (mm)
        - rel_x: distance forward from robot
        - rel_y: distance left from robot
    """
    # Create SE(2) transformation for robot pose in world frame
    T_world_robot = SE2(robot_x, robot_y, robot_theta)

    # Get inverse transformation (robot frame to world frame)
    T_robot_world = T_world_robot.inv()

    # Transform point from world frame to robot frame
    world_point = np.array([world_x, world_y])
    rel_point = T_robot_world * world_point

    return rel_point[0].item(), rel_point[1].item()


def relative_to_world(
    rel_x: float,
    rel_y: float,
    robot_x: float,
    robot_y: float,
    robot_theta: float
) -> Tuple[float, float]:
    """
    Convert robot-relative coordinates to world coordinates using SE(2).

    Args:
        rel_x: Distance forward from robot (mm)
        rel_y: Distance left from robot (mm)
        robot_x: Robot x position in world frame (mm)
        robot_y: Robot y position in world frame (mm)
        robot_theta: Robot orientation in world frame (radians, 0 = facing forward/+x)

    Returns:
        Tuple of (world_x, world_y) in world frame (mm)
    """
    # Create SE(2) transformation for robot pose in world frame
    T_world_robot = SE2(robot_x, robot_y, robot_theta)

    # Transform point from robot frame to world frame
    rel_point = np.array([rel_x, rel_y], dtype=np.float64)
    world_point = T_world_robot * rel_point

    return world_point[0].item(), world_point[1].item()


def transform_pose_world_to_relative(
    world_x: float,
    world_y: float,
    world_theta: float,
    robot_x: float,
    robot_y: float,
    robot_theta: float
) -> Tuple[float, float, float]:
    """
    Convert full pose (position + orientation) from world to robot-relative frame using SE(2).

    Args:
        world_x: Target x position in world frame (mm)
        world_y: Target y position in world frame (mm)
        world_theta: Target orientation in world frame (radians)
        robot_x: Robot x position in world frame (mm)
        robot_y: Robot y position in world frame (mm)
        robot_theta: Robot orientation in world frame (radians)

    Returns:
        Tuple of (rel_x, rel_y, rel_theta) in robot-relative frame
    """
    # Create SE(2) transformations
    T_world_robot = SE2(robot_x, robot_y, robot_theta)
    T_world_target = SE2(world_x, world_y, world_theta)

    # Compute relative transformation:
    T_robot_target = T_world_robot.inv() * T_world_target

    # Extract pose (x, y, theta)
    rel_x = T_robot_target.t[0]
    rel_y = T_robot_target.t[1]
    rel_theta = T_robot_target.theta()

    return rel_x.item(), rel_y.item(), float(rel_theta)


def transform_pose_relative_to_world(
    rel_x: float,
    rel_y: float,
    rel_theta: float,
    robot_x: float,
    robot_y: float,
    robot_theta: float
) -> Tuple[float, float, float]:
    """
    Convert full pose (position + orientation) from robot-relative to world frame using SE(2).

    Args:
        rel_x: Distance forward from robot (mm)
        rel_y: Distance left from robot (mm)
        rel_theta: Orientation relative to robot (radians)
        robot_x: Robot x position in world frame (mm)
        robot_y: Robot y position in world frame (mm)
        robot_theta: Robot orientation in world frame (radians)

    Returns:
        Tuple of (world_x, world_y, world_theta) in world frame
    """
    # Create SE(2) transformations
    T_world_robot = SE2(robot_x, robot_y, robot_theta)
    T_robot_target = SE2(rel_x, rel_y, rel_theta)

    # Compute world transformation: T_world_target = T_world_robot *
    # T_robot_target
    T_world_target = T_world_robot * T_robot_target

    # Extract pose (x, y, theta)
    world_x = T_world_target.t[0]
    world_y = T_world_target.t[1]
    world_theta = T_world_target.theta()

    return world_x.item(), world_y.item(), float(world_theta)

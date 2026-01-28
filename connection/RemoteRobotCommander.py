"""
Remote robot command sender - sends commands to the Pi over TCP.

This module handles sending movement and gripper commands to the robot
over a TCP socket connection.
"""

import socket
from typing import Optional, Tuple
from . import protocol
from . import message_types
from IRobotCommander import IRobotCommander  # type: ignore


class RemoteRobotCommander(IRobotCommander):
    """
    Remote command sender for robot control over network.

    Sends movement and gripper commands to the Raspberry Pi over TCP.
    """

    def __init__(self, command_socket: Optional[socket.socket] = None):
        """
        Initialize the remote commander.

        Args:
            command_socket: TCP socket for sending commands (can be set later)
        """
        super().__init__()
        self.command_socket = command_socket

    def set_socket(self, command_socket: Optional[socket.socket]) -> None:
        """
        Set or update the command socket.

        Args:
            command_socket: TCP socket for sending commands
        """
        self.command_socket = command_socket

    def close(self) -> bool:
        """
        Send close command to the Pi to gracefully shut down the connection.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        return protocol.send_command(
            self.command_socket, message_types.CLOSE, [])

    def early_game(
            self,
            golden: Tuple[float, float],
            left: Tuple[float, float],
            right: Tuple[float, float]) -> bool:
        """
        Send early game strategy command with can locations.

        Args:
            golden: (x, y) coordinates of golden can in mm
            left: (x, y) coordinates of left can in mm
            right: (x, y) coordinates of right can in mm

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        args = [golden[0], golden[1], left[0], left[1], right[0], right[1]]
        return protocol.send_command(
            self.command_socket,
            message_types.EARLY_GAME,
            args
        )

    def override_movement(self, movement_args: list[float]) -> bool:
        """
        Send list of movement commands to the Pi.

        Args:
            movement_args: list of movement commands in groups of 3
                left_coef: Left motor coefficient (-1.0 to 1.0)
                right_coef: Right motor coefficient (-1.0 to 1.0)
                distance: Distance to move (in ticks)

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        assert (len(movement_args) % 3 == 0)

        return protocol.send_command(
            self.command_socket,
            message_types.OVERRIDE_MOVEMENTS,
            movement_args
        )

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
        if not self.command_socket:
            return False

        assert (len(movement_args) % 2 == 0)

        return protocol.send_command(
            self.command_socket,
            message_types.OVERRIDE_WAYPOINTS,
            movement_args
        )

    def override_relative_xy(self, x: float, y: float) -> bool:
        """
        Override current path with relative movement in ROS coordinates.

        Args:
            x: Forward distance in mm (positive = forward, negative = backward)
            y: Lateral distance in mm (positive = left, negative = right)

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print(f'Sending relative xy: x={x}, y={y}')
        return protocol.send_command(
            self.command_socket,
            message_types.OVERRIDE_RELATIVE_XY,
            [x, y]
        )

    def override_world_xy(self, world_x: float, world_y: float) -> bool:
        """
        Override current path to navigate to world coordinates.

        Args:
            world_x: Target x position in world frame (mm)
            world_y: Target y position in world frame (mm)

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print(f'Sending world coordinates: x={world_x}, y={world_y}')
        return protocol.send_command(
            self.command_socket,
            message_types.OVERRIDE_WORLD_XY,
            [world_x, world_y]
        )

    def pickup_can(self) -> bool:
        """
        Pick up a can with the gripper.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print('Sending pickup can command')
        return protocol.send_command(
            self.command_socket,
            message_types.PICKUP_CAN,
            []
        )

    def release_can(self) -> bool:
        """
        Release the can from the gripper.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print('Sending release can command')
        return protocol.send_command(
            self.command_socket,
            message_types.RELEASE_CAN,
            []
        )

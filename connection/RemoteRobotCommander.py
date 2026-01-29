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
        self.next_command_id = 1

    def set_socket(self, command_socket: Optional[socket.socket]) -> None:
        """
        Set or update the command socket.

        Args:
            command_socket: TCP socket for sending commands
        """
        self.command_socket = command_socket

    def _get_command_id(self) -> int:
        """Get the next command ID and increment the counter."""
        command_id = self.next_command_id
        self.next_command_id += 1
        self.last_command_id = command_id
        return command_id

    def get_last_command_id(self) -> int:
        """Get the ID of the last command sent (for waiting on completion)."""
        return getattr(self, 'last_command_id', 0)

    def close(self) -> bool:
        """
        Send close command to the Pi to gracefully shut down the connection.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        return protocol.send_command(
            self.command_socket, message_types.CLOSE, [], command_id=0)

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
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.EARLY_GAME,
            args,
            command_id=command_id
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

        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.OVERRIDE_MOVEMENTS,
            movement_args,
            command_id=command_id
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

        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.OVERRIDE_WAYPOINTS,
            movement_args,
            command_id=command_id
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
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.OVERRIDE_RELATIVE_XY,
            [x, y],
            command_id=command_id
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
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.OVERRIDE_WORLD_XY,
            [world_x, world_y],
            command_id=command_id
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
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.PICKUP_CAN,
            [],
            command_id=command_id
        )

    def pickup_tipped_can(self) -> bool:
        """
        Pick up a tipped-over can with the gripper.

        Note:
            The current network protocol does not distinguish between
            tipped and upright cans, so this uses the same message type
            as ``pickup_can``.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print('Sending pickup tipped can command (alias for pickup can)')
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.PICKUP_CAN,
            [],
            command_id=command_id
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
            [],
            command_id=0
        )

    def open_gripper(self) -> bool:
        """
        Open the gripper to prepare for grabbing.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print('Sending open gripper command')
        return protocol.send_command(
            self.command_socket,
            message_types.OPEN_GRIPPER,
            [],
            command_id=0
        )

    def lower_elevator(self) -> bool:
        """
        Lower the elevator mechanism.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print('Sending lower elevator command')
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.LOWER_ELEVATOR,
            [],
            command_id=command_id
        )

    def approach_can_with_ds(self, max_iterations: int = 100) -> bool:
        """
        Approach can using distance sensor feedback.

        Uses distance sensor to approach can in real-time, stopping when
        within 20mm or returning False if no can detected or max iterations exceeded.

        Args:
            max_iterations: Maximum number of loop iterations before giving up (default 100)

        Returns:
            True if successfully approached can, False if no can detected or max exceeded
        """
        if not self.command_socket:
            return False

        print(f'Sending approach can with distance sensor command (max_iterations={max_iterations})')
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.APPROACH_CAN_DS,
            [float(max_iterations)],
            command_id=command_id
        )

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
        if not self.command_socket:
            return False

        print(
            f'Sending stack command: temp_pos={temp_pos}, stack_pos={stack_pos}, stacked_cans={stacked_cans}')
        args = [
            temp_pos[0],
            temp_pos[1],
            stack_pos[0],
            stack_pos[1],
            float(stacked_cans)]
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.STACK,
            args,
            command_id=command_id
        )

    def waitFinishedMoving(self) -> bool:
        """
        Wait for current movement to complete.

        Blocks until the robot's navigation system reports no movement.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print('Sending wait movement finished command')
        return protocol.send_command(
            self.command_socket,
            message_types.WAIT_MOVEMENT_FINISHED,
            [],
            command_id=0
        )

    def reset_gripper(self) -> bool:
        """
        Reset the gripper servo.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print('Sending reset gripper command')
        return protocol.send_command(
            self.command_socket,
            message_types.RESET_GRIPPER,
            [],
            command_id=0
        )

    def set_down_can(self) -> bool:
        """
        Set down a can at the current position.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print('Sending set down can command')
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.SET_DOWN_CAN,
            [],
            command_id=command_id
        )

    def backup(self) -> bool:
        """
        Back up the robot a short distance.

        Returns:
            True if successful
        """
        if not self.command_socket:
            return False

        print('Sending backup command')
        command_id = self._get_command_id()
        return protocol.send_command(
            self.command_socket,
            message_types.BACKUP,
            [],
            command_id=command_id
        )

    def complete_command_immediately(self, command_id: int) -> None:
        """
        Mark a command as complete.
        Not used on computer side (Pi handles completion).

        Args:
            command_id: The ID of the command that completed
        """
        pass

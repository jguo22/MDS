import math
import numpy as np
from enum import Enum, auto
from typing import Tuple, List
from spatialmath import SE2
from IRobotCommander import IRobotCommander  # type: ignore
from connection.frame_info import FrameInfo
from navHelpers import get_rotate
from vision.segment import segmentImage
from vision.can_utils import getCans
from vision.relativeCoordinates import relative_to_world, world_to_relative
from profiler import Profiler
from thetaStar import ThetaStar
from streamer import Streamer
from config import FPS, CAN_DIAMETER, ROBOT_DIAMETER, APPROACH_OFFSET
from colors import GREEN_ZONE, RED_ZONE, GOLDEN_ZONE, GREEN_CAN, RED_CAN, GOLDEN_CAN, canNamesToNumbers


class RobotStateSimple(Enum):
    SearchForCan = auto()
    MoveToCan = auto()
    GrabCan = auto()
    MoveToZone = auto()
    Done = auto()


class RobotHandlerSimple:
    """Simplified robot handler with basic can collection strategy."""

    # Constants
    X_CENTER_BORDER = 3000  # Exclude cans with x >= this value

    def __init__(self, robot_commander: IRobotCommander):
        self.state = RobotStateSimple.SearchForCan
        self.started = False
        self.paused = False
        self.waiting_for_command_id = 0

        # Configuration
        self.MAX_STACK = 3  # Number of cans to collect
        self.cans_collected = 0  # Total cans delivered to zone
        self.cans_held = 0  # Number of cans currently holding
        self.target_zone = RED_ZONE
        self.target_can_color = GREEN_CAN  # Color of cans to search for

        # Memory
        # Hardcoded zone centers for testing
        self.zones: list[np.ndarray] = []
        self.zones[GREEN_ZONE] = np.array([[1670, 584]])
        self.zones[RED_ZONE] = np.array([[990, -939.8]])
        self.zones[GOLDEN_ZONE] = np.array([[2235, -914]])
        self.cans: List[Tuple[float, float]] = []
        self.can_colors: List[int] = []
        self.can_detections: dict[Tuple[int, int], int] = {}
        self.DETECTION_THRESHOLD = 3

        # Current state
        self.current_can: Tuple[float, float, int] = (0, 0, -1)
        self.claw_raised = True
        self.can_in_gripper = False

        # Robot components
        self.robot_commander = robot_commander
        self.thetaStar = ThetaStar()
        self.profiler = Profiler(False)
        self.telemetry = Streamer()

        # Frame data
        self.frame_top: np.ndarray = np.array([[]])
        self.frame_bottom: np.ndarray = np.array([[]])
        self.frame_id = -1
        self.robot_pose = SE2(0, 0, 0)

        # Segmentation results
        self.result_top = None
        self.result_bottom = None

        self.telemetry.set_data(self.get_picklable_dict())

        self.cans_collected = 0
        self.cans_held = 0
        self.claw_raised = True
        self.can_in_gripper = False
        self.state = RobotStateSimple.SearchForCan

    def handleFrame(self, frame_info: FrameInfo):
        self.profiler.start_frame()

        # if self.paused:
        #     return
        # self.robot_commander.open_gripper()
        # self.robot_commander.lower_elevator()
        # self.paused = True
        # return

        if (not self.started) or self.paused:
            self.profiler.end_frame()
            return

        # Check if waiting for command
        if self.waiting_for_command_id > 0:
            if frame_info.lastCompletedCommandId >= self.waiting_for_command_id:
                print(f"Command {self.waiting_for_command_id} completed")
                self.waiting_for_command_id = 0
            else:
                self.profiler.end_frame()
                return

        # Update frame data
        self.frame_top = frame_info.frame_top
        self.frame_bottom = frame_info.frame_bottom
        self.frame_id = frame_info.frame_id
        self.robot_pose = SE2(frame_info.x, frame_info.y, frame_info.theta)

        # Run segmentation
        self.result_top = segmentImage(self.frame_top)
        self.result_bottom = segmentImage(self.frame_bottom)
        self.profiler.record("segmentImage")

        # Update can detections
        self.updateCanDetections()

        self.profiler.record("scanAndSetZones")

        # State machine
        if self.state == RobotStateSimple.SearchForCan:
            self.handleSearchForCan()
        elif self.state == RobotStateSimple.MoveToCan:
            self.handleMoveToCan()
        elif self.state == RobotStateSimple.GrabCan:
            self.handleGrabCan()
        elif self.state == RobotStateSimple.MoveToZone:
            self.handleMoveToZone()
        elif self.state == RobotStateSimple.Done:
            print("Task complete!")

        self.profiler.record("handleState")
        self.updateTelemetry()
        self.profiler.record("telemetry")
        self.profiler.end_frame()

    def handleSearchForCan(self):
        """Rotate until a can of target color is detected."""
        # Filter cans to only target color
        target_cans = self.getTargetColorCans()

        if len(target_cans) == 0:
            # Keep rotating
            rotate_cmd = list(get_rotate(math.pi / 3 / FPS))
            # print(
            # f"Searching for can {self.cans_collected + 1}/{self.MAX_STACK}")
            self.robot_commander.override_movement(rotate_cmd)
        else:
            print(f"Can found! Moving to it...")
            self.state = RobotStateSimple.MoveToCan

    def handleMoveToCan(self):
        """Navigate to the nearest can of target color."""
        # Filter cans to only target color
        target_cans = self.getTargetColorCans()
        print("navigating")

        if len(target_cans) == 0:
            print("No cans of target color found, searching again...")
            self.state = RobotStateSimple.SearchForCan
            return
        print("can found")

        # Get closest target can
        robot_x, robot_y, theta = unpackPose(self.robot_pose)
        closest_idx = 0
        closest_dist = float('inf')
        for i, (can_x, can_y, can_color) in enumerate(target_cans):
            dist = getDistance((robot_x, robot_y), (can_x, can_y))
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        can_x, can_y, can_color = target_cans[closest_idx]
        print(can_x, can_y)

        # Check if close enough to approach
        if self.isPointClose(can_x, can_y):
            print(f"Reached can at ({can_x:.0f}, {can_y:.0f})")
            self.current_can = (can_x, can_y, can_color)
            # Remove this can from the list
            for i, (cx, cy) in enumerate(self.cans):
                if getDistance((cx, cy), (can_x, can_y)) < 10:
                    self.cans.pop(i)
                    self.can_colors.pop(i)
                    break
            self.robot_commander.approach_can_with_ds()
            if self.can_in_gripper:
                self.robot_commander.open_gripper()
                self.robot_commander.lower_elevator()
            self.robot_commander.pickup_can()
            self.waiting_for_command_id = self.robot_commander.get_last_command_id()
            self.state = RobotStateSimple.GrabCan
        else:
            # Move closer
            self.thetaStarAndSend(can_x, can_y)

    def handleGrabCan(self):
        """Wait for grab to complete and update state."""
        # Wait for previous commands to finish
        self.robot_commander.waitFinishedMoving()

        self.cans_held += 1
        self.can_in_gripper = True
        self.claw_raised = True

        print(f"Can grabbed! Holding {self.cans_held}/{self.MAX_STACK} cans")

        # Check if done collecting
        if self.cans_held >= self.MAX_STACK:
            print(f"Holding {self.MAX_STACK} cans, moving to zone...")
            self.state = RobotStateSimple.MoveToZone
        else:
            print(f"Searching for next can...")
            self.state = RobotStateSimple.SearchForCan

    def handleMoveToZone(self):
        """Move to target zone and drop cans."""
        zone = self.zones[self.target_zone]
        if zone is None:
            print("Zone not found, searching...")
            return

        zx, zy = zone

        # Check if at zone
        if self.isPointClose(zx, zy):
            print(f"Reached zone at ({zx:.0f}, {zy:.0f})")
            print(f"Dropping {self.cans_held} cans...")
            self.robot_commander.set_down_can()
            self.robot_commander.backup()
            self.cans_collected += self.cans_held
            self.cans_held = 0
            self.waiting_for_command_id = self.robot_commander.get_last_command_id()
            print(f"Total cans delivered: {self.cans_collected}")
            self.state = RobotStateSimple.Done
        else:
            self.thetaStarAndSend(zx, zy)

    def updateCanDetections(self) -> None:
        """Detect cans from segmentation and keep old detections."""
        all_locations: List[Tuple[float, float]] = []
        all_colors: List[int] = []

        for result, frame, is_top in [
            (self.result_top, self.frame_top, True),
            (self.result_bottom, self.frame_bottom, False)
        ]:
            locations, color_strings = getCans(result, frame, is_top)
            colors = canNamesToNumbers(color_strings)

            # Transform to world coordinates
            locations = [relative_to_world(location, self.robot_pose)
                         for location in locations]

            all_locations.extend(locations)
            all_colors.extend(colors)

        # Update detection tracking
        new_detections: dict[Tuple[int, int], int] = {}

        for i, location in enumerate(all_locations):
            rounded = (
                round(location[0] / 50) * 50,
                round(location[1] / 50) * 50
            )

            matched = False
            for existing_loc, count in self.can_detections.items():
                if getDistance(rounded, existing_loc) < CAN_DIAMETER:
                    new_detections[existing_loc] = count + 1
                    matched = True
                    break

            if not matched:
                new_detections[rounded] = 1

        # Build confirmed cans list
        confirmed_cans: List[Tuple[float, float]] = []
        confirmed_colors: List[int] = []

        # Add newly confirmed cans
        for rounded_loc, count in new_detections.items():
            if count >= self.DETECTION_THRESHOLD:
                for i, location in enumerate(all_locations):
                    rounded = (
                        round(location[0] / 50) * 50,
                        round(location[1] / 50) * 50
                    )
                    if rounded == rounded_loc:
                        confirmed_cans.append(location)
                        confirmed_colors.append(all_colors[i])
                        break

        # Keep old cans that are still valid
        for old_can, old_color in zip(self.cans, self.can_colors):
            # Skip if already in new confirmed list
            already_added = False
            for new_can in confirmed_cans:
                if getDistance(old_can, new_can) < CAN_DIAMETER / 2:
                    already_added = True
                    break

            if not already_added:
                confirmed_cans.append(old_can)
                confirmed_colors.append(old_color)

        self.can_detections = new_detections
        self.cans = confirmed_cans
        self.can_colors = confirmed_colors

    # Helper functions

    def getTargetColorCans(self) -> List[Tuple[float, float, int]]:
        """Get list of cans matching target color and within bounds."""
        target_cans = []
        for i, (can_x, can_y) in enumerate(self.cans):
            if self.can_colors[i] == self.target_can_color and can_x < self.X_CENTER_BORDER:
                target_cans.append((can_x, can_y, self.can_colors[i]))
        return target_cans

    def isPointClose(self, x: float, y: float) -> bool:
        """Check if point is close enough for straight movement."""
        local_x, local_y = world_to_relative((x, y), self.robot_pose)

        distance = math.sqrt(local_x**2 + local_y**2)
        if distance <= ROBOT_DIAMETER / 2:
            return True

        rect_length = APPROACH_OFFSET + 20
        rect_width = CAN_DIAMETER
        in_rectangle = (
            0 <= local_x <= rect_length and
            -rect_width / 2 <= local_y <= rect_width / 2
        )

        return in_rectangle

    def getWorldClawOffsetPosition(
            self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Get position offset by claw length to approach can properly."""
        dx, dy = world_to_relative(point, self.robot_pose)
        gx, gy = relative_to_world(
            (max(dx - APPROACH_OFFSET + 10, 0), dy), self.robot_pose)
        return gx, gy

    def thetaStarAndSend(self, x: float, y: float):
        """Plan path using theta* and send to robot."""
        # Get offset position to approach from correct distance
        goal_x, goal_y = self.getWorldClawOffsetPosition((x, y))
        print(x, y)
        print(goal_x, goal_y)

        robot_x, robot_y, theta = unpackPose(self.robot_pose)
        self.thetaStar.set_start(robot_x, robot_y)
        self.thetaStar.set_goal(goal_x, goal_y)
        waypoints = self.thetaStar.path_find()

        command_args = [robot_x, robot_y]
        for wx, wy in waypoints:
            command_args.append(wx)
            command_args.append(wy)
        print(command_args)
        self.robot_commander.override_waypoints(command_args)

    def updateTelemetry(self):
        """Update telemetry data for visualization."""
        scaling = 0.001
        x, y, theta = unpackPose(self.robot_pose)
        self.telemetry.update_odom_state(x * scaling, y * scaling, theta)

        circles = []
        for i, (cx, cy) in enumerate(self.cans):
            circles.append((cx * scaling, cy * scaling, "blue"))
        self.telemetry.update_circles(circles)

        data = self.get_picklable_dict()
        self.telemetry.set_data(data)

    def get_picklable_dict(self):
        """Get picklable state dict."""
        exclude = {
            'robot_commander', 'thetaStar', 'profiler', 'telemetry',
            'frame_bottom', 'frame_top', 'robot_pose',
            'result_top', 'result_bottom'
        }

        result = {}
        for k, v in self.__dict__.items():
            if k not in exclude:
                if isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                elif isinstance(v, dict):
                    safe_dict = {}
                    for dk, dv in v.items():
                        safe_key = ",".join(
                            str(x) for x in dk) if isinstance(
                            dk, tuple) else dk
                        safe_dict[safe_key] = dv
                    result[k] = safe_dict
                elif isinstance(v, list):
                    result[k] = [
                        item.tolist() if isinstance(
                            item, np.ndarray) else item for item in v]
                else:
                    result[k] = v

        result['robot_pose'] = [
            self.robot_pose.x,
            self.robot_pose.y,
            self.robot_pose.theta()]
        return result


def getDistance(point1: Tuple[float, float], point2: Tuple[float, float]):
    x1, y1 = point1
    x2, y2 = point2
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def unpackPose(pose: SE2) -> Tuple[float, float, float]:
    x = float(pose.x)
    y = float(pose.y)
    theta = pose.theta()
    if type(theta) is float:
        return x, y, theta
    else:
        return x, y, theta[0]

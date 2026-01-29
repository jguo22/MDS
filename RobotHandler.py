import math
import numpy as np
from enum import Enum, auto
from typing import Tuple, List
from spatialmath import SE2
from torch.nn.modules import distance
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


class RobotState(Enum):
    SearchForCan = auto()
    MoveToCan = auto()
    ApproachingCan = auto()
    PickupCan = auto()
    MoveToZone = auto()
    Done = auto()


class RobotHandler:
    """Simplified robot handler with basic can collection strategy."""

    # Constants
    def __init__(self, robot_commander: IRobotCommander):
        self.state = RobotState.SearchForCan
        self.started = False
        self.paused = False
        self.waiting_for_command_id = 0

        # Configuration
        self.MAX_STACK = 2  # Number of cans to collect
        self.cans_collected = 0  # Total cans delivered to zone
        self.cans_held = 0  # Number of cans currently holding
        self.target_zone = GREEN_ZONE
        self.target_can_color = RED_CAN  # Color of cans to search for

        # Memory
        # Hardcoded zone centers for testing
        self.zones: list[Tuple[float, float]] = [(0, 0)] * 3
        # HERE
        self.zones[GREEN_ZONE] = (1000, 0)
        self.zones[RED_ZONE] = (990, -939.8)
        self.zones[GOLDEN_ZONE] = (2235, -914)
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
        self.state = RobotState.SearchForCan

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
        self.distanceSensed = frame_info.distanceSensed

        # Run segmentation
        self.result_top = segmentImage(self.frame_top)
        self.result_bottom = segmentImage(self.frame_bottom)
        self.profiler.record("segmentImage")

        # Update can detections
        self.updateCanDetections()

        self.profiler.record("scanAndSetZones")

        # State machine
        if self.state == RobotState.SearchForCan:
            self.handleSearchForCan()
        elif self.state == RobotState.MoveToCan:
            self.handleMoveToCan()
        elif self.state == RobotState.ApproachingCan:
            self.handleApproachingCan()
        elif self.state == RobotState.PickupCan:
            self.handlePickupCan()
        elif self.state == RobotState.MoveToZone:
            self.handleMoveToZone()
        elif self.state == RobotState.Done:
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
            self.robot_commander.override_movement(rotate_cmd)
        else:
            print(f"Can found! Moving to it...")
            self.state = RobotState.MoveToCan

    def handleMoveToCan(self):
        """Navigate to the nearest can of target color."""
        # Filter cans to only target color
        target_cans = self.getTargetColorCans()
        print("navigating")

        if len(target_cans) == 0:
            print("No cans of target color found, searching again...")
            self.state = RobotState.SearchForCan
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
            print(
                f"Close to can at ({can_x:.0f}, {can_y:.0f}), starting approach")
            self.current_can = (can_x, can_y, can_color)
            # Remove this can from the list
            for i, (cx, cy) in enumerate(self.cans):
                if getDistance((cx, cy), (can_x, can_y)) < 10:
                    self.cans.pop(i)
                    self.can_colors.pop(i)
                    break
            self.state = RobotState.ApproachingCan
        else:
            # Move closer
            self.thetaStarAndSend(can_x, can_y)
            self.waiting_for_command_id = self.robot_commander.get_last_command_id()

    def handleApproachingCan(self):
        """Use distance sensor to approach can, limited iterations per frame."""
        print("Approaching can with distance sensor...")

        # Do one iteration of approach
        print("approachig with ds")
        if self.distanceSensed <= 25:
            self.handlePickupCan()
        else:
            self.robot_commander.approach_can_with_ds()
            self.waiting_for_command_id = self.robot_commander.get_last_command_id()

    def handlePickupCan(self):
        # Check if we're close enough to grab
        # This will be checked next frame after approach completes
        # For now, assume approach worked and proceed to grab
        if self.can_in_gripper:
            print("Opening gripper for new can")
            self.robot_commander.open_gripper()
            self.robot_commander.lower_elevator()

        print("Picking up can")
        self.robot_commander.pickup_can()
        self.waiting_for_command_id = self.robot_commander.get_last_command_id()

        self.cans_held += 1
        self.can_in_gripper = True
        self.claw_raised = True

        print(f"Can grabbed! Holding {self.cans_held}/{self.MAX_STACK} cans")

        # Check if done collecting
        if self.cans_held >= self.MAX_STACK:
            print(f"Holding {self.MAX_STACK} cans, moving to zone...")
            self.state = RobotState.MoveToZone
        else:
            print(f"Searching for next can...")
            self.state = RobotState.SearchForCan

    def handleMoveToZone(self):
        """Move to target zone and drop cans."""
        zone = self.zones[self.target_zone]
        if zone is None:
            print("Zone not found, searching...")
            return

        print(zone)
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
            self.state = RobotState.Done
        else:
            self.thetaStarAndSend(zx, zy)

    def updateCanDetections(self) -> None:
        """Detect cans from segmentation and update tracking."""
        GRID_SIZE = 50  # mm - grid for matching detections
        MATCH_DISTANCE = CAN_DIAMETER  # Distance to consider same can

        # Get all detections from both cameras
        current_detections: List[Tuple[float, float]] = []
        current_colors: List[int] = []

        for result, frame, is_top in [
            (self.result_top, self.frame_top, True),
            (self.result_bottom, self.frame_bottom, False)
        ]:
            locations, color_strings = getCans(result, frame, is_top)
            colors = canNamesToNumbers(color_strings)

            # Transform to world coordinates
            for loc, color in zip(locations, colors):
                world_loc = relative_to_world(loc, self.robot_pose)
                current_detections.append(world_loc)
                current_colors.append(color)

        # Update detection counts (for stability filtering)
        new_detection_counts: dict[Tuple[int, int], int] = {}

        for location in current_detections:
            grid_loc = (
                round(location[0] / GRID_SIZE) * GRID_SIZE,
                round(location[1] / GRID_SIZE) * GRID_SIZE
            )

            # Check if matches existing tracked location
            matched_key = None
            for existing_loc in self.can_detections:
                if getDistance(grid_loc, existing_loc) < MATCH_DISTANCE:
                    matched_key = existing_loc
                    break

            if matched_key:
                new_detection_counts[matched_key] = self.can_detections[matched_key] + 1
            else:
                new_detection_counts[grid_loc] = 1

        # Build final can list
        final_cans: List[Tuple[float, float]] = []
        final_colors: List[int] = []

        # Add cans that meet detection threshold
        for grid_loc, count in new_detection_counts.items():
            if count >= self.DETECTION_THRESHOLD:
                # Find the actual detection closest to this grid location
                best_loc = None
                best_color = -1
                best_dist = float('inf')

                for loc, color in zip(current_detections, current_colors):
                    dist = getDistance(loc, grid_loc)
                    if dist < best_dist:
                        best_dist = dist
                        best_loc = loc
                        best_color = color

                if best_loc and best_dist < MATCH_DISTANCE:
                    # Check not already added
                    is_duplicate = False
                    for existing in final_cans:
                        if getDistance(existing, best_loc) < MATCH_DISTANCE:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        final_cans.append(best_loc)
                        final_colors.append(best_color)

        # Keep old confirmed cans that weren't re-detected
        for old_can, old_color in zip(self.cans, self.can_colors):
            is_duplicate = False
            for new_can in final_cans:
                if getDistance(old_can, new_can) < MATCH_DISTANCE:
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_cans.append(old_can)
                final_colors.append(old_color)

        self.can_detections = new_detection_counts
        self.cans = final_cans
        self.can_colors = final_colors

    # Helper functions
    def getTargetColorCans(self) -> List[Tuple[float, float, int]]:
        """Get list of cans matching target color and within bounds."""
        target_cans = []
        # TODO: put back color color
        # for i, (can_x, can_y) in enumerate(self.cans):
        #     if self.can_colors[i] == self.target_can_color and isWithinBounds(
        #             can_x, can_y):
        #         target_cans.append((can_x, can_y, self.can_colors[i]))
        for i, (can_x, can_y) in enumerate(self.cans):
            if isWithinBounds(can_x, can_y):
                target_cans.append((can_x, can_y, self.can_colors[i]))
        return target_cans

    def isPointClose(self, x: float, y: float) -> bool:
        """Check if point is close enough for straight movement."""
        local_x, local_y = world_to_relative((x, y), self.robot_pose)

        distance = math.sqrt(local_x**2 + local_y**2)
        if distance <= ROBOT_DIAMETER / 2:
            return True

        rect_length = APPROACH_OFFSET + 70
        rect_width = CAN_DIAMETER * 2
        in_rectangle = (
            0 <= local_x <= rect_length and
            -rect_width / 2 <= local_y <= rect_width / 2
        )

        return in_rectangle

    def getWorldClawOffsetPosition(
            self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Get position offset by claw length to approach can properly."""
        robot_x, robot_y, _ = unpackPose(self.robot_pose)
        dx = point[0] - robot_x
        dy = point[1] - robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < APPROACH_OFFSET:
            return robot_x, robot_y

        scale = (dist - APPROACH_OFFSET) / dist
        gx = robot_x + dx * scale
        gy = robot_y + dy * scale

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

        # Filter out waypoints less than 5mm from start
        waypoints = [
            (wx, wy) for wx, wy in waypoints
            if getDistance((robot_x, robot_y), (wx, wy)) >= 5.0
        ]

        command_args = [robot_x, robot_y]
        for wx, wy in waypoints:
            command_args.append(wx)
            command_args.append(wy)
        print(command_args)
        self.robot_commander.override_waypoints(command_args)
        self.robot_commander.waitFinishedMoving()
        self.waiting_for_command_id = self.robot_commander.get_last_command_id()

    def updateTelemetry(self):
        scaling = 0.001
        x, y, theta = unpackPose(self.robot_pose)
        self.telemetry.update_odom_state(x * scaling, y * scaling, theta)

        circles = []
        for i in range(len(self.cans)):
            cx, cy = self.cans[i]
            cx *= scaling
            cy *= scaling
            color = self.can_colors[i]
            if color == GREEN_CAN:
                circles.append((cx, cy, "green"))
            elif color == RED_CAN:
                circles.append((cx, cy, "red"))
            elif color == GOLDEN_CAN:
                circles.append((cx, cy, "gold"))
        for i in range(len(self.zones)):
            circles.append((self.zones[i][0], self.zones[i][1], "white"))
        self.telemetry.update_circles(circles)

        data = self.get_picklable_dict()
        self.telemetry.set_data(data)

    def get_picklable_dict(self):
        """Get picklable state dict."""
        exclude = {
            'robot_commander', 'thetaStar', 'profiler', 'telemetry',
            'frame_bottom', 'frame_top', 'robot_pose',
            'result_top', 'result_bottom', 'state'
        }

        result = {}
        result['state'] = self.state.name
        result['robot_pose'] = [
            self.robot_pose.x,
            self.robot_pose.y,
            self.robot_pose.theta()]
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


low_x = 0
high_x = 3048
low_y = 0
high_y = 3048
x_offset = 600
y_offset = 304.8


def isWithinBounds(x, y):
    return x + x_offset > low_x and x + x_offset < high_x and y + \
        y_offset > low_y and y + y_offset < high_y

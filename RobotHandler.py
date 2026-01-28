import time
import math
import numpy as np
from enum import Enum, auto
from typing import Tuple, List, Optional
from spatialmath import SE2
from IRobotCommander import IRobotCommander  # type: ignore
from connection.frame_info import FrameInfo
from navHelpers import get_rotate
from vision.segment import segmentImage
from vision.zone_utils import doPolygonsIntersect, getSquareCenter, getZones, isPointInPoly
from vision.can_utils import getCans
from vision.relativeCoordinates import relative_to_world, world_to_relative
from profiler import Profiler
from thetaStar import ThetaStar
from streamer import Streamer
from config import FPS, CAN_DIAMETER, BASE_D, CLAW_OFFSET, ROBOT_DIAMETER, SCOOPER_LENGTH, TEMP_STACK_OFFSET
from colors import GREEN_CAN, GREEN_ZONE, GREEN_ZONE_OPP, RED_CAN, RED_ZONE, RED_ZONE_OPP, GOLDEN_CAN, GOLDEN_ZONE, GOLDEN_ZONE_OPP, ZONE_CLASS_NAMES, canNamesToNumbers


class RobotState(Enum):
    StartScan = auto()
    StartGather = auto()
    MidgameSearch = auto()
    MidgameDecide = auto()
    MidgameGoToCan = auto()
    MidgameGrabbing = auto()
    MidgameGoToZone = auto()
    MidgameStacking = auto()
    FinishedStacking = auto()


class RobotHandler():
    def __init__(self, robot_commander: IRobotCommander):
        # state variables
        self.state = RobotState.StartScan
        self.started = False
        self.paused = False
        # Command ID we're waiting for (0 = not waiting)
        self.waiting_for_command_id = 0

        # BEST GUESS MEMORY VARIABLES
        # four vertices of scoring zones in world coords
        # list of zones, each zone is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]
        self.zones: List[Optional[np.ndarray]] = [
            None, None, None, None, None, None]
        self.zone_confidences = [0, 0, 0, 0, 0, 0]
        # Store planned path to cans
        self.cans: List[Tuple[float, float]] = []
        self.can_colors: List[int] = []
        # Number of consecutive frames each can has been visible but not detected
        # Aligned by index with self.cans / self.can_colors
        self.can_miss_counts: List[int] = []
        # x, y, stack size, color, id
        self.stacked_cans: List[Tuple[float, float, int, int, int]] = []
        self.borders: List[Tuple[int, int]] = []

        # VARIABLES FOR CURRENT STATE
        # can_x, can_y, can_color
        self.current_can: Tuple[float, float, int] = (0, 0, -1)
        # zone to go to
        self.targetZone: int = -1
        # target of stacking state: stack id
        self.targetStackId: int = -1
        self.lastStackId: int = -1

        # random
        self.startTime = time.time()
        self.startFrame: int = -1
        self.lastTimeSentPath = 0

        self.robot_commander = robot_commander
        self.thetaStar = ThetaStar()
        self.profiler = Profiler(False)
        self.telemetry = Streamer()

        # information from current frame
        self.frame_top: np.ndarray = np.array([[]])
        self.frame_bottom: np.ndarray = np.array([[]])
        self.frame_id = -1
        self.robot_pose = SE2(0, 0, 0)
        self.distanceSensed = 0
        self.didEarlyGame = False

        # Segmentation results for visualization
        self.result_top = None
        self.result_bottom = None

        self.telemetry.set_data(self.get_picklable_dict())

    def start(self):
        self.startFrame = -1
        self.startTime = time.time()

    def handleFrame(self, frame_info: FrameInfo):
        self.profiler.start_frame()

        # Use top camera frame for vision processing
        self.frame_top = frame_info.frame_top
        self.frame_bottom = frame_info.frame_bottom
        self.frame_id = frame_info.frame_id
        self.robot_pose = SE2(frame_info.x, frame_info.y, frame_info.theta)
        self.distanceSensed = frame_info.distanceSensed

        # Skip processing if paused
        if self.paused:
            self.profiler.end_frame()
            return

        # Check if we're waiting for a command to complete
        if self.waiting_for_command_id > 0:
            # Check if the command we're waiting for has completed
            if frame_info.lastCompletedCommandId >= self.waiting_for_command_id:
                # Command completed, clear waiting state
                print(
                    f"on frame {self.frame_id}, waiting command id {self.waiting_for_command_id} finished")
                self.waiting_for_command_id = 0
            else:
                # Still waiting, skip state processing
                self.profiler.end_frame()
                return

        self.result_top = segmentImage(self.frame_top)
        self.result_bottom = segmentImage(self.frame_bottom)
        self.profiler.record("segmentImage")

        for result, frame, is_top in [
            (self.result_top, self.frame_top, True),
            (self.result_bottom, self.frame_bottom, False)
        ]:
            self.scanAndSetZones(result, frame, is_top)
            locations, color_strings = getCans(result, frame)
            colors = canNamesToNumbers(color_strings)

            locations = [relative_to_world(location, self.robot_pose)
                         for location in locations]

            # Ensure miss-count list is aligned with cans list
            if len(self.can_miss_counts) != len(self.cans):
                self.can_miss_counts = [0] * len(self.cans)

            # Start new lists with currently detected cans (miss count = 0)
            new_locations: List[Tuple[float, float]] = list(locations)
            new_colors: List[int] = list(colors)
            new_miss_counts: List[int] = [0] * len(new_locations)

            # Check each old can to see if it should be kept or removed
            for i in range(len(self.cans)):
                old_can_x, old_can_y = self.cans[i]
                old_color = self.can_colors[i]
                miss_count = self.can_miss_counts[i]

                # Check if old can matches any new detection
                has_nearby_detection = any(
                    getDistance(self.cans[i], locations[j]) < CAN_DIAMETER / 2
                    for j in range(len(locations))
                )

                if has_nearby_detection:
                    # Already represented by a current detection (miss count reset via detection)
                    continue

                # No nearby detection for this old can
                if self.is_world_point_visible(old_can_x, old_can_y, is_top):
                    # Visible in FOV but not detected this frame
                    miss_count += 1
                    # Only remove if it has been visible and undetected for 5 consecutive frames
                    if miss_count >= 5:
                        continue  # Drop this stale can
                # If not visible, keep the existing miss_count (do not increment)

                # Keep the can (either not visible or not yet past miss threshold)
                new_locations.append(self.cans[i])
                new_colors.append(old_color)
                new_miss_counts.append(miss_count)

            self.cans = new_locations
            self.can_colors = new_colors
            self.can_miss_counts = new_miss_counts

            # TESTING PURPOSES
            self.zones[GREEN_ZONE] = np.array([[918.62, 288.33],
                                               [922.48, -271.63],
                                               [1391.22, -262.14],
                                               [1382.95, 269.04]])
            self.zone_confidences[GREEN_ZONE] = 2
            self.zones[RED_ZONE] = np.array([[2071.79, -26.68],
                                             [1791.50, 311.42],
                                             [1438.28, 7.33],
                                             [1710.01, -324.33]])
            self.zone_confidences[RED_ZONE] = 2
            self.zones[GOLDEN_ZONE] = np.array([[1896.03, -681.89],
                                                [1832.41, -610.53],
                                                [1732.2, -675.07],
                                                [1811.42, -762.24]])
            self.zone_confidences[GOLDEN_ZONE] = 2

        self.profiler.record("scanAndSetZones")

        # Dispatch to appropriate state handler
        if self.state == RobotState.StartScan:
            self.handleStartScan(self.frame_id)
        elif self.state == RobotState.StartGather:
            self.handleStartGather()
        elif self.state == RobotState.MidgameSearch:
            self.handleMidgameSearch()
        elif self.state == RobotState.MidgameDecide:
            self.handleMidgameDecide()
        elif self.state == RobotState.MidgameGoToCan:
            self.handleMidgameGoToCan()
        elif self.state == RobotState.MidgameGrabbing:
            self.handleMidgameGrabbing()
        elif self.state == RobotState.MidgameGoToZone:
            self.handleMidgameGoToZone()
        elif self.state == RobotState.MidgameStacking:
            self.handleMidgameStacking()
        elif self.state == RobotState.FinishedStacking:
            self.handleFinishedStacking()

        self.profiler.record("handleState")

        self.updateTelemetry()
        self.profiler.record("telemetry")

        time.sleep(0)
        self.profiler.record("sleep")

        self.profiler.end_frame()

    # ------------------------ STATE FUNCTIONS .----------------------------

    def handleStartScan(self, frame_id: int):
        """Handle StartScan state: detect cans and plan initial path"""
        self.state = RobotState.StartScan
        if self.startFrame == -1:
            self.startFrame = frame_id
            self.robot_commander.reset_gripper()

        if self.started:
            self.state = RobotState.StartGather

    def handleStartGather(self):
        """Handle StartGather state: send waypoints and check if cans reached"""
        self.state = RobotState.StartGather

        self.targetZone = GREEN_ZONE
        self.handleMidgameGoToZone()
        return
        # ---------- SEND PATH IF IT HASN'T BEEN SENT YET -------------
        # if time.time() - self.lastTimeSentPath > 100:
        #     self.lastTimeSentPath = time.time()
        #
        #     self.send_waypoints(self.cans)
        #
        # while len(self.cans) > 0 and self.isPointInScooper(*self.cans[0]):
        #     self.cans.pop(0)

        # Sort cans by y value, keeping colors aligned
        if not self.didEarlyGame:
            self.didEarlyGame = True

            sorted_pairs = sorted(
                zip(self.cans, self.can_colors), key=lambda pair: -pair[0][1])
            sorted_cans = [can for can, color in sorted_pairs]
            sorted_colors = [color for can, color in sorted_pairs]

            # Find the golden can
            golden_can = None
            for i, color in enumerate(sorted_colors):
                if color == GOLDEN_CAN:
                    golden_can = sorted_cans[i]
                    break
            if golden_can is None:
                golden_can = sorted_cans[len(sorted_cans) // 2]

            self.robot_commander.send_early_game(
                golden_can, sorted_cans[0], sorted_cans[-1])

    def handleMidgameSearch(self):
        """Handle Search state: rotate slowly until target zone is found"""
        self.state = RobotState.MidgameSearch
        # Check if target zone has been found
        if self.zones[self.targetZone] is None:
            # Rotate slowly (45 degrees every second)
            rotate_cmd = list(get_rotate(math.pi / 4 / FPS))
            self.robot_commander.override_movement(rotate_cmd)
        else:
            self.state = RobotState.MidgameGoToZone

    def handleMidgameDecide(self):
        """Handle Midgame state: placeholder for midgame logic"""
        self.state = RobotState.MidgameGrabbing
        print("NO MORE CANS. PLACEHOLDER")

    def handleMidgameGoToCan(self):
        """
        Handle MidgameGoToCan state: navigate to a can
        """
        self.state = RobotState.MidgameGoToCan
        if len(self.cans) == 0:
            self.handleMidgameDecide()
            return

        # Sort cans by distance from robot (nearest first)
        robot_x, robot_y = self.robot_pose.x, self.robot_pose.y
        robot_pos = (robot_x, robot_y)
        sorted_pairs = sorted(
            zip(self.cans, self.can_colors),
            key=lambda pair: getDistance(robot_pos, pair[0])
        )
        self.cans = [can for can, color in sorted_pairs]
        self.can_colors = [color for can, color in sorted_pairs]

        # this allows for dynamic update of which can to go to
        can_x, can_y = self.cans[0]
        can_color = self.can_colors[0]
        if self.isPointClose(can_x, can_y):
            # remove it from list cuz its gonna get moved
            self.cans.pop(0)

            # logic for grabbing it
            self.current_can = (can_x, can_y, can_color)
            print("grabbing can at ")
            print(can_x, can_y)
            self.handleMidgameGrabbing()
        else:
            # move to can using theta*
            # dx, dy = world_to_relative((can_x, can_y), self.robot_pose)
            # gx, gy = relative_to_world((max(dx - 200, 0), dy), self.robot_pose)
            # self.robot_commander.override_world_xy(gx, gy)
            # self.robot_commander.waitFinishedMoving()
            # print("real positions and goals")
            # print(can_x, can_y)
            # print(gx, gy)
            self.thetaStarAndSend(can_x, can_y)

    def handleMidgameGrabbing(self):
        """
        Grab a can that is in the scooper
        """
        self.state = RobotState.MidgameGrabbing

        cx, cy, can_color = self.current_can
        if can_color not in [GREEN_CAN, RED_CAN, GOLDEN_CAN]:
            self.state = RobotState.MidgameGoToCan
            return

        if self.isPointClose(cx, cy):
            self.robot_commander.approach_can_with_ds()
            self.robot_commander.pickup_can()
            self.waiting_for_command_id = self.robot_commander.get_last_command_id()
            print(
                f"on frame {self.frame_id}, sent a waiting command with id {self.waiting_for_command_id}")

            # Select target zone based on can color
            if can_color == GREEN_CAN:
                self.targetZone = GREEN_ZONE
            elif can_color == RED_CAN:
                self.targetZone = RED_ZONE
            elif can_color == GOLDEN_CAN:
                self.targetZone = GOLDEN_ZONE

            # TODO: logic to see if we actually got it
            self.state = RobotState.MidgameGoToZone
        else:
            self.thetaStarAndSend(cx, cy)
            self.state = RobotState.MidgameGoToCan

    def handleMidgameGoToZone(self):
        """Handle MidgameGoToZone state: navigate to zone and release can"""
        self.state = RobotState.MidgameGoToZone

        # Check if robot is in target zone
        if self.zones[self.targetZone] is None:
            self.handleMidgameSearch()
            return

        self.targetStackId = -1
        goal = None
        for stack in self.stacked_cans:
            x, y, size, color, id = stack
            if size == 0:
                print("WARNING: stack size of 0")
                continue
            if color == self.targetZone:
                if isPointInPoly((x, y), self.zones[self.targetZone]):
                    goal = (x, y)
                    self.targetStackId = id
                    break
                else:
                    print("WARNING: CAN STACK NOT IN ZONE")
        if goal is None:
            goal = getSquareCenter(self.zones[self.targetZone])
        print("go to zone goal")
        print(goal)

        if self.isPointClose(*goal):
            if self.targetStackId != 0:
                self.robot_commander.approach_can_with_ds()
                self.robot_commander.pickup_can()
                self.waiting_for_command_id = self.robot_commander.get_last_command_id()
                print(
                    f"on frame {self.frame_id}, sent a waiting command with id {self.waiting_for_command_id}")
            self.handleMidgameStacking()
            print("stacking")
        else:
            self.thetaStarAndSend(*goal)

    def handleMidgameStacking(self):
        """
            Once the robot is already holding a can
            Call the function to stack it in one superframe
        """
        self.state = RobotState.MidgameStacking

        # see if there a target stack
        targetStack = None
        for stack in self.stacked_cans:
            stackId = stack[4]
            if self.targetStackId == stackId:
                targetStack = stack
                break

        # otherwise, create a new stack of 0
        if targetStack is None:
            x, y = getSquareCenter(self.zones[self.targetZone])
            targetStack = (x, y, 0, self.targetZone, self.lastStackId + 1)
            self.lastStackId += 1

        # get the target
        cx, cy, color, height, id = targetStack

        # calculate offsetted position
        zone_x, zone_y = getSquareCenter(self.zones[self.targetZone])
        dx = zone_x - cx
        dy = zone_y - cy
        distance = math.sqrt(dx * dx + dy * dy)

        # Move 200mm towards zone center (or less if zone center is closer)
        if distance > 0:
            temp_pos = (cx + dx / distance * TEMP_STACK_OFFSET,
                        cy + dy / distance * TEMP_STACK_OFFSET)
        else:
            # Stack is already at zone center, offset in x direction
            temp_pos = (cx + TEMP_STACK_OFFSET, cy)

        stack_pos = (cx, cy)
        self.robot_commander.stack(temp_pos, stack_pos, height)
        self.waiting_for_command_id = self.robot_commander.get_last_command_id()
        print(
            f"on frame {self.frame_id}, sent a waiting command with id {self.waiting_for_command_id}")

        self.state = RobotState.FinishedStacking

    def handleFinishedStacking(self):
        # get position of stacked cans, which should be right in front after
        # stacking
        cx, cy = relative_to_world(
            (CLAW_OFFSET + CAN_DIAMETER / 2, 0), self.robot_pose)

        # update list of stacked cans
        for i in range(len(self.stacked_cans)):
            _x, _y, color, prev_height, id = self.stacked_cans[i]
            if self.targetStackId == id:
                self.stacked_cans[i] = (cx, cy, color, prev_height + 1, id)
                break

        self.thetaStar.addCan(cx, cy)

        self.robot_commander.override_movement([-1, -1, 3000])
        self.waiting_for_command_id = self.robot_commander.get_last_command_id()
        print(
            f"on frame {self.frame_id}, sent a waiting command with id {self.waiting_for_command_id}")
        self.state = RobotState.MidgameGoToCan

    # ------------------------ HELPER FUNCTIONS .----------------------------

    def scanAndSetZones(self, result, image, is_top):
        """
        Detects and assigns the 6 scoring zones from YOLO results.
        Only updates zones that haven't been detected yet (are None).

        Args:
            result: YOLO result object from inference
            image: Original BGR image used for zone detection
        """
        # Get zones sorted by distance (closest first)
        squares_xy, class_names, confidences = getZones(
            result, image, is_top)

        # Iterate through all detected zones
        for zone, name, conf in zip(squares_xy, class_names, confidences):
            if name == ZONE_CLASS_NAMES[GREEN_ZONE]:
                self.updateZone(zone, conf, GREEN_ZONE, GREEN_ZONE_OPP)

            elif name == ZONE_CLASS_NAMES[RED_ZONE]:
                self.updateZone(zone, conf, RED_ZONE, RED_ZONE_OPP)

            elif name == ZONE_CLASS_NAMES[GOLDEN_ZONE]:
                self.updateZone(zone, conf, GOLDEN_ZONE, GOLDEN_ZONE_OPP)

    def updateZone(self, zone, conf, our_zone_id, their_zone_id):
        prev_zone = self.zones[our_zone_id]
        prev_conf = self.zone_confidences[their_zone_id]
        if prev_zone is None:
            self.zones[our_zone_id] = zone
            self.zone_confidences[our_zone_id] = conf
        else:
            if doPolygonsIntersect(prev_zone, zone):
                # if they intersect, they're probably detecting the
                # same zone and use the one thats better
                if conf > self.zone_confidences[our_zone_id]:
                    self.zones[our_zone_id] = zone
                    self.zone_confidences[our_zone_id] = conf
            else:
                # TODO: figure out which zone is ours
                # using actual logic
                prev_x, prev_y = getSquareCenter(prev_zone)
                prevDistSquared = prev_x * prev_x + prev_y * prev_y
                curr_x, curr_y = getSquareCenter(zone)
                currDistSquared = curr_x * curr_x + curr_y * curr_y
                if currDistSquared < prevDistSquared:
                    # current zone is our zone and other zone might be
                    # others
                    if prev_conf > self.zone_confidences[their_zone_id]:
                        self.zones[their_zone_id] = prev_zone
                        self.zone_confidences[their_zone_id] = prev_conf
                    self.zones[our_zone_id] = zone
                    self.zone_confidences[our_zone_id] = conf
                else:
                    # current zone might be theirs
                    if conf > self.zone_confidences[their_zone_id]:
                        self.zones[their_zone_id] = zone
                        self.zone_confidences[their_zone_id] = conf

    def isPointInScooper(self, x: float, y: float) -> bool:
        """
        Check if a point is within reach of the robot (circular radius OR forward rectangle).

        Uses SE(2) transformation to convert to robot-relative coordinates, then checks:
        1. Within circular radius: (BASE_D - CAN_DIAMETER) / 2
        2. Within rectangle: SCOOPER_LENGTH forward, (BASE_D - CAN_DIAMETER) wide

        Args:
            x: x world coordinate in mm
            y: y world coordinate in mm

        Returns:
            bool: True if point is reachable
        """

        # Transform point to robot's local coordinate frame using SE2
        local_x, local_y = world_to_relative((x, y), self.robot_pose)

        # Check 1: Is point within circular radius?
        distance = math.sqrt(local_x**2 + local_y**2)
        if distance <= (BASE_D - CAN_DIAMETER) / 2:
            return True

        # Check 2: Is point within rectangle in front of robot?
        # Check if point is within rectangle bounds
        # Rectangle extends from 0 to rect_length in front (local_x)
        # and from -rect_width/2 to +rect_width/2 sideways (local_y)
        rect_length = SCOOPER_LENGTH
        rect_width = BASE_D - CAN_DIAMETER
        in_rectangle = (
            0 <= local_x <= rect_length and
            -rect_width / 2 <= local_y <= rect_width / 2
        )

        return in_rectangle

    def isPointInGripper(self, x: float, y: float) -> bool:
        """
        Check if a point is within reach of the robot (circular radius OR forward rectangle).

        Uses SE(2) transformation to convert to robot-relative coordinates, then checks:
        1. Within circular radius: (BASE_D - CAN_DIAMETER) / 2
        2. Within rectangle: SCOOPER_LENGTH forward, (BASE_D - CAN_DIAMETER) wide

        Args:
            x: x world coordinate in mm
            y: y world coordinate in mm

        Returns:
            bool: True if point is reachable
        """
        local_x, local_y = world_to_relative((x, y), self.robot_pose)

        # Check 1: Would the can be fully in the robot?
        # Handles measurement inaccuracy
        distance = math.sqrt(local_x**2 + local_y**2)
        if distance <= (BASE_D - CAN_DIAMETER) / 2:
            return True

        # Check 2: Is point within rectangle in front of robot?
        rect_length = CLAW_OFFSET
        rect_width = BASE_D - CAN_DIAMETER
        in_rectangle = (
            0 <= local_x <= rect_length and
            -rect_width / 2 <= local_y <= rect_width / 2
        )

        return in_rectangle

    def isPointClose(self, x: float, y: float) -> bool:
        """
        Check if a point is close enough so that we can move straight forward
        without bumping anything in between the robot and can
        """
        local_x, local_y = world_to_relative((x, y), self.robot_pose)

        # Check 1: Would the can be fully in the robot?
        # Handles measurement inaccuracy
        distance = math.sqrt(local_x**2 + local_y**2)
        if distance <= ROBOT_DIAMETER / 2:
            return True

        # Check 2: Is point within rectangle in front of robot?
        rect_length = CLAW_OFFSET + CAN_DIAMETER * 3 / 2
        rect_width = CAN_DIAMETER
        in_rectangle = (
            0 <= local_x <= rect_length and
            -rect_width / 2 <= local_y <= rect_width / 2
        )

        return in_rectangle

    def is_world_point_visible(self, world_x: float, world_y: float, is_top: bool) -> bool:
        """
        Check if a world point is visible in the camera's field of view.

        Args:
            world_x: x coordinate in world frame (mm)
            world_y: y coordinate in world frame (mm)
            is_top: True for top camera, False for bottom camera

        Returns:
            True if the point is visible in the specified camera's FOV
        """
        from vision.pixelTo3D import H_TOP, H_BOTTOM
        from vision.relativeCoordinates import world_to_pixel
        from config import FRAME_WIDTH, FRAME_HEIGHT

        # Convert world coordinates to robot-relative coordinates
        camera_relative = world_to_relative((world_x, world_y), self.robot_pose)

        # Points behind the camera cannot be visible
        if camera_relative[0] < 0:
            return False

        # Get the appropriate homography matrix
        h_matrix = H_TOP if is_top else H_BOTTOM

        # Try to project to pixel coordinates
        pixel_coords = world_to_pixel(camera_relative, h_matrix)
        if pixel_coords is None:
            return False

        # Check if pixel coordinates are within frame bounds
        u, v = pixel_coords
        return 0 <= u < FRAME_WIDTH and 0 <= v < FRAME_HEIGHT

    def send_waypoints(self, waypoints: List[Tuple[float, float]]):
        x, y, theta = unpackPose(self.robot_pose)
        command_args = [x, y]
        for x, y in waypoints:
            command_args.append(x)
            command_args.append(y)
        self.robot_commander.override_waypoints(command_args)

    def thetaStarAndSend(self, x: float, y: float):
        # temporary while thetastar doesn't work
        # dx, dy = world_to_relative((x, y), self.robot_pose)
        # gx, gy = relative_to_world((max(dx - 150, 0), dy), self.robot_pose)
        # self.robot_commander.override_world_xy(gx, gy)
        self.robot_commander.override_world_xy(x, y)
        self.waiting_for_command_id = self.robot_commander.get_last_command_id()
        print(
            f"on frame {self.frame_id}, sent a waiting command with id {self.waiting_for_command_id}")
        print("real positions and goals")
        print(x, y)
        # print(gx, gy)
        #
        # robot_x = self.robot_pose.x
        # robot_y = self.robot_pose.y
        #
        # self.thetaStar.set_start(robot_x, robot_y)
        # self.thetaStar.set_goal(x, y)
        # waypoints = self.thetaStar.path_find()
        #
        # self.send_waypoints(waypoints)

    def updateTelemetry(self):
        # self.telemetry.set_img(cv2.Mat(self.frame_top))
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
            else:
                print("INVALID COLOR")
        self.telemetry.update_circles(circles)

        # Plot zones as lines
        lines = []
        zone_colors = {
            GREEN_ZONE: "green",
            RED_ZONE: "red",
            GOLDEN_ZONE: "gold",
            GREEN_ZONE_OPP: "lightgreen",
            RED_ZONE_OPP: "pink",
            GOLDEN_ZONE_OPP: "yellow"
        }

        for zone_id, zone in enumerate(self.zones):
            if zone is not None:
                color = zone_colors.get(zone_id, "white")
                # Draw 4 lines connecting the vertices in a closed loop
                for i in range(4):
                    x1, y1 = zone[i]
                    x2, y2 = zone[(i + 1) %
                                  4]  # Wrap around to close the polygon
                    lines.append((x1 * scaling, y1 * scaling,
                                 x2 * scaling, y2 * scaling, color))

        self.telemetry.update_lines(lines)

        data = self.get_picklable_dict()
        self.telemetry.set_data(data)

    def get_picklable_dict(self):
        """Returns a dict with unpicklable objects removed and enums converted to strings."""

        exclude = {
            'robot_commander',
            'thetaStar',
            'profiler',
            'telemetry',
            'frame_bottom',
            'frame_top',
            'robot_pose',
            'result_top',
            'result_bottom',
        }

        result = {}
        for k, v in self.__dict__.items():
            if k in exclude:
                continue

            # Convert enum to string name
            if isinstance(v, Enum):
                result[k] = v.name
            # Convert numpy arrays to lists
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            # Convert lists of numpy arrays to lists of lists
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


def getDistance(point1, point2):
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

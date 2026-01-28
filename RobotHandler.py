import time
import math
import numpy as np
from enum import Enum, auto
from typing import Tuple, List
from spatialmath import SE2
from IRobotCommander import IRobotCommander  # type: ignore
from connection.frame_info import FrameInfo
from navHelpers import get_rotate
import navHelpers
from vision.segment import segmentImage
from vision.zone_utils import doPolygonsIntersect, getSquareCenter, getZones, isPointInPoly
from vision.can_utils import getCans
from vision.relativeCoordinates import relative_to_world, world_to_relative
from profiler import Profiler
from thetaStar import ThetaStarPlanner
from streamer import Streamer
from config import FPS, CAN_HEIGHT, CAN_DIAMETER, BASE_D, CLAW_OFFSET, SCOOPER_LENGTH
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


class RobotHandler():
    def __init__(self, robot_commander: IRobotCommander):
        # state variables
        self.state = RobotState.StartScan
        self.started = False

        # BEST GUESS MEMORY VARIABLES
        # four vertices of scoring zones in world coords
        # np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        self.zones = [None, None, None, None, None, None]
        self.zone_confidences = [0, 0, 0, 0, 0, 0]
        # Store planned path to cans
        self.cans: List[Tuple[float, float]] = []
        self.can_colors: List[int] = []
        # x, y, stack size, color
        self.stacked_cans: List[Tuple[float, float, int, int]] = []
        self.borders: List[Tuple[int, int]] = []

        # VARIABLES FOR CURRENT STATE
        # can_x, can_y, can_color
        self.current_can: Tuple[float, float, int] = (0, 0, -1)
        # zone to go to
        self.targetZone: int = -1
        # target of stacking state: x, y, height
        self.targetStack: Tuple[float, float, float] = (0, 0, 0)

        # random
        self.startTime = time.time()
        self.startFrame: int = -1
        self.lastTimeSentPath = 0

        self.robot_commander = robot_commander
        self.thetaStar = ThetaStarPlanner()
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

        result_top = segmentImage(self.frame_top)
        # result_bottom = segmentImage(self.frame_bottom)

        # Store results for visualization
        self.result_top = result_top
        # self.result_bottom = result_bottom

        self.profiler.record("segmentImage")

        for result, frame, is_top in [
            (result_top, self.frame_top, True),
            # (result_bottom, self.frame_bottom, False)
        ]:
            self.scanAndSetZones(result, frame, is_top)
            locations, color_strings = getCans(result, frame)
            colors = canNamesToNumbers(color_strings)

            locations = [relative_to_world(location, self.robot_pose)
                         for location in locations]

            for i in range(len(self.cans)):
                not_repeat = True
                for j in range(len(locations)):
                    if distance(
                            self.cans[i],
                            locations[j]) < CAN_DIAMETER:
                        not_repeat = False
                        break
                if not_repeat:
                    locations.append(self.cans[i])
                    colors.append(self.can_colors[i])

            self.cans = locations
            self.can_colors = colors

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
        else:
            print("ERROR: INVALID STATE")

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

        if self.started:
            self.state = RobotState.StartGather

    def handleStartGather(self):
        """Handle StartGather state: send waypoints and check if cans reached"""
        self.state = RobotState.StartGather

        self.handleMidgameGoToCan()
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

        # this allows for dynamic update of which can to go to
        can_x, can_y = self.cans[0]
        can_color = self.can_colors[0]
        if self.isPointClose(can_x, can_y):
            # remove it from list cuz its gonna get moved
            self.cans.pop(0)

            # logic for grabbing it
            self.current_can = (can_x, can_y, can_color)
            self.handleMidgameGrabbing()
        else:
            # move to can using theta*
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

        if self.isPointInGripper(cx, cy):
            self.robot_commander.override_movement([])
            self.robot_commander.pickup_can()

            # Select target zone based on can color
            if can_color == GREEN_CAN:
                self.targetZone = GREEN_ZONE
            elif can_color == RED_CAN:
                self.targetZone = RED_ZONE
            elif can_color == GOLDEN_CAN:
                self.targetZone = GOLDEN_ZONE

            self.state = RobotState.MidgameGoToZone
        elif self.isPointClose(cx, cy):
            self.preciseMoveToTarget(cx, cy)
        else:
            self.thetaStarAndSend(cx, cy)
            self.state = RobotState.MidgameGoToCan

    def handleMidgameGoToZone(self):
        """Handle MidgameGoToZone state: navigate to zone and release can"""
        self.state = RobotState.MidgameGoToZone

        # Check if robot is in target zone
        if self.zones[self.targetZone] is None:
            self.handleMidgameSearch()
            self.state = RobotState.MidgameSearch
            return

        goal = None
        stack_size = 0
        for x, y, size, color in self.stacked_cans:
            if color == self.targetZone:
                if isPointInPoly((x, y), self.zones[self.targetZone]):
                    goal = (x, y)
                    stack_size = size
                else:
                    print("WARNING: CAN STACK NOT IN ZONE")
        if goal is None:
            goal = getSquareCenter(self.zones[self.targetZone])

        if self.isPointClose(*goal):
            self.handleMidgameStacking()
        else:
            self.thetaStarAndSend(*goal)

    def handleMidgameStacking(self):
        """Handle MidgameStacking state"""
        self.state = RobotState.MidgameStacking

        cx, cy, height = self.targetStack

        if self.isPointInGripper(cx, cy):
            self.robot_commander.release_can()
        elif self.isPointClose(cx, cy):
            self.preciseMoveToTarget(cx, cy)
        else:
            self.thetaStarAndSend(cx, cy)
            self.state = RobotState.MidgameGoToZone

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
        if distance <= (BASE_D - CAN_DIAMETER) / 2:
            return True

        # Check 2: Is point within rectangle in front of robot?
        rect_length = CLAW_OFFSET + CAN_DIAMETER * 3 / 2
        rect_width = CAN_DIAMETER
        in_rectangle = (
            0 <= local_x <= rect_length and
            -rect_width / 2 <= local_y <= rect_width / 2
        )

        return in_rectangle

    def send_waypoints(self, waypoints: List[Tuple[float, float]]):
        x = self.robot_pose.x
        y = self.robot_pose.y
        args = [x, y]
        for point in waypoints:
            args.append(point[0])
            args.append(point[1])

        self.robot_commander.override_waypoints(args)

    def thetaStarAndSend(self, x: float, y: float):
        robot_x = self.robot_pose.x
        robot_y = self.robot_pose.y
        self.thetaStar.build_map(robot_x, robot_y, x, y)
        rx, ry = self.thetaStar.planning(robot_x, robot_y, x, y)
        waypoints = list(zip(rx, ry))
        self.send_waypoints(waypoints)

    def preciseMoveToTarget(self, gx, gy):
        # TODO: add better logic
        distanceToMove = min(self.distanceSensed - CAN_DIAMETER / 2, 0)
        movement_args = list(navHelpers.get_forward_mm(distanceToMove))
        self.robot_commander.override_movement(movement_args)

    def updateTelemetry(self):
        # self.telemetry.set_img(cv2.Mat(self.frame_top))
        scaling = 0.001
        x = self.robot_pose.x
        y = self.robot_pose.y
        theta = self.robot_pose.theta()
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


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)

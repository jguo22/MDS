import time
import math
import cv2
import numpy as np
from enum import Enum, auto
from typing import Tuple, List
from spatialmath import SE2
from connection.ComputerReceiver import ComputerReceiver
from connection.frame_info import FrameInfo
from navHelpers import get_forward_mm, get_rotate
import navHelpers
from yolo.segment import segmentImage
from yolo.zone_utils import getSquareCenter, getZones, isPointInPoly
from yolo.can_utils import getCans
from coordinates.relativeCoordinates import get_movement_plan, world_to_relative
from profiler import Profiler
from thetaStar import ThetaStarPlanner
from streamer import Streamer
from config import FPS, CAN_HEIGHT, CENTER_BORDER_X, CAN_DIAMETER, BASE_D, CLAW_OFFSET, SCOOPER_LENGTH
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
    def __init__(self, computer_receiver: ComputerReceiver):
        # state variables
        self.state = RobotState.StartScan
        self.started = False

        # BEST GUESS MEMORY VARIABLES
        # four vertices of scoring zones in world coords
        # np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        self.zones = [None, None, None, None, None, None]
        # Store planned path to cans
        self.cans: List[Tuple[float, float]] = []
        self.can_colors: List[int] = []
        # x, y, stack size, color
        self.stacked_cans: List[Tuple[float, float, int, int]] = []

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

        self.computer_receiver = computer_receiver
        self.thetaStar = ThetaStarPlanner()
        self.profiler = Profiler()
        self.telemetry = Streamer()

        # information from current frame
        self.frame_top: np.ndarray = np.array([[]])
        self.frame_bottom: np.ndarray = np.array([[]])
        self.frame_id = -1
        self.robot_pose = SE2(0, 0, 0)
        self.distanceSensed = 0

        print(self.get_picklable_dict())
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

        result = segmentImage(self.frame_top)
        self.profiler.record("segmentImage")

        # scan and set any zones that haven't been found yet
        self.scanAndSetZones(result, self.frame_top)
        self.profiler.record("scanAndSetZones")

        # Dispatch to appropriate state handler
        if self.state == RobotState.StartScan:
            self.handleStartScan(self.frame_id, result, self.frame_top)
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

        self.telemetry.set_img(cv2.Mat(self.frame_top))
        self.telemetry.update_odom_state(
            frame_info.x, frame_info.y, frame_info.theta)
        circles = []
        for i in range(len(self.cans)):
            cx, cy = self.cans[i]
            color = self.can_colors[i]
            if color == GREEN_CAN:
                circles.append((cx, cy, "green"))
            if color == RED_CAN:
                circles.append((cx, cy, "red"))
            if color == GOLDEN_CAN:
                circles.append((cx, cy, "gold"))
        self.telemetry.update_circles(circles)
        self.telemetry.set_data(self.get_picklable_dict())

        self.profiler.record("telemetry")

        self.profiler.end_frame()

    # ------------------------ STATE FUNCTIONS .----------------------------

    def handleStartScan(self, frame_id: int, result, frame: np.ndarray):
        """Handle StartScan state: detect cans and plan initial path"""
        self.state = RobotState.StartScan
        if self.startFrame == -1:
            self.startFrame = frame_id

            # ------------- PLAN PATH TO DETECTED CANS -------------
            # Get all detected cans in image coordinates
            can_locations_xy, can_colors = getCans(result, frame)

            # Store the planned path
            self.cans = can_locations_xy
            self.can_colors = canNamesToNumbers(can_colors)

            if self.started:
                self.state = RobotState.StartGather

    def handleStartGather(self):
        """Handle StartGather state: send waypoints and check if cans reached"""
        self.state = RobotState.StartGather
        # ---------- SEND PATH IF IT HASN'T BEEN SENT YET -------------
        if time.time() - self.lastTimeSentPath > 2:
            self.lastTimeSentPath = time.time()

            self.send_waypoints(self.cans)

        while len(self.cans) > 0 and self.isPointInScooper(*self.cans[0]):
            self.cans.pop(0)

    def handleMidgameSearch(self):
        """Handle Search state: rotate slowly until target zone is found"""
        self.state = RobotState.MidgameSearch
        # Check if target zone has been found
        if self.zones[self.targetZone] is None:
            # Rotate slowly (45 degrees every second)
            rotate_cmd = list(get_rotate(math.pi / 4 / FPS))
            self.computer_receiver.override_movement(rotate_cmd)
        else:
            self.state = RobotState.MidgameGoToZone

    def handleMidgameDecide(self):
        """Handle Midgame state: placeholder for midgame logic"""
        self.state = RobotState.MidgameGrabbing

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
            self.computer_receiver.override_movement([])
            self.computer_receiver.send_grip_can(CAN_HEIGHT)
            # TODO: add logic to check if its done picking up

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
            self.computer_receiver.send_gripper_height(stack_size * CAN_HEIGHT)
            self.thetaStarAndSend(*goal)

    def handleMidgameStacking(self):
        """Handle MidgameStacking state"""
        self.state = RobotState.MidgameStacking

        cx, cy, height = self.targetStack

        if self.isPointInGripper(cx, cy):
            self.computer_receiver.send_release_can(height)
        elif self.isPointClose(cx, cy):
            self.preciseMoveToTarget(cx, cy)
        else:
            self.thetaStarAndSend(cx, cy)
            self.state = RobotState.MidgameGoToZone

    # ------------------------ HELPER FUNCTIONS .----------------------------

    def scanAndSetZones(self, result, image):
        """
        Detects and assigns the 6 scoring zones from YOLO results.
        Only updates zones that haven't been detected yet (are None).
        Uses x-coordinate to determine ours vs opponent: x < CENTER_BORDER_X is ours.

        Args:
            result: YOLO result object from inference
            image: Original BGR image used for zone detection
        """
        # Get zones sorted by distance (closest first)
        squares_xy, class_names = getZones(result, image)

        # Iterate through all detected zones
        for quad, name in zip(squares_xy, class_names):
            # Calculate center x-coordinate to determine which side
            center_x = np.mean(quad[:, 0])
            is_our_side = center_x < CENTER_BORDER_X

            if name == ZONE_CLASS_NAMES[GREEN_ZONE]:
                if is_our_side and self.zones[GREEN_ZONE] is None:
                    self.zones[GREEN_ZONE] = quad
                elif not is_our_side and self.zones[GREEN_ZONE_OPP] is None:
                    self.zones[GREEN_ZONE_OPP] = quad

            elif name == ZONE_CLASS_NAMES[RED_ZONE]:
                if is_our_side and self.zones[RED_ZONE] is None:
                    self.zones[RED_ZONE] = quad
                elif not is_our_side and self.zones[RED_ZONE_OPP] is None:
                    self.zones[RED_ZONE_OPP] = quad

            elif name == ZONE_CLASS_NAMES[GOLDEN_ZONE]:
                if is_our_side and self.zones[GOLDEN_ZONE] is None:
                    self.zones[GOLDEN_ZONE] = quad
                elif not is_our_side and self.zones[GOLDEN_ZONE_OPP] is None:
                    self.zones[GOLDEN_ZONE_OPP] = quad

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
        plan = get_movement_plan(waypoints, self.robot_pose)

        movement_args = []
        for move in plan:
            dist, theta = move
            movement_args.extend(get_rotate(theta))
            movement_args.extend(get_forward_mm(dist))

        self.computer_receiver.override_movement(movement_args)
        self.lastTimeSentPath = time.time()

    def thetaStarAndSend(self, x: float, y: float):
        robot_x, robot_y, theta = self.robot_pose
        rx, ry = self.thetaStar.planning(robot_x, robot_y, x, y)
        waypoints = list(zip(rx, ry))
        self.send_waypoints(waypoints)

    def preciseMoveToTarget(self, gx, gy):
        # TODO: add better logic
        distanceToMove = min(self.distanceSensed - CAN_DIAMETER / 2, 0)
        movement_args = list(navHelpers.get_forward_mm(distanceToMove))
        self.computer_receiver.override_movement(movement_args)

    def get_picklable_dict(self):
        """Returns a dict with unpicklable objects removed and enums converted to strings."""

        exclude = {
            'computer_receiver',
            'thetaStar',
            'profiler',
            'telemetry',
        }

        result = {}
        for k, v in self.__dict__.items():
            if k in exclude:
                continue

            # Convert enum to string name
            if isinstance(v, Enum):
                result[k] = v.name
            else:
                result[k] = v

        return result

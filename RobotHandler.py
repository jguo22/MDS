import time
import math
import numpy as np
from enum import Enum, auto
from typing import Tuple, List
from spatialmath import SE2
from connection.ComputerReceiver import ComputerReceiver
from navHelpers import get_forward_mm, get_rotate
from yolo.segment import segmentImage
from yolo.zone_utils import getQuadCenter, getZones, isPointInPoly
from yolo.can_utils import getCans
from config import CAN_HEIGHT, CENTER_BORDER_X, CAN_DIAMETER, BASE_D, CLAW_OFFSET, FT_TO_MM, SCOOPER_LENGTH
from coordinates.relativeCoordinates import get_movement_plan, world_to_relative
from profiler import Profiler
from thetaStar import ThetaStarPlanner
from config import FPS


class RobotState(Enum):
    StartScan = auto()
    StartGather = auto()
    MidgameSearch = auto()
    MidgameDecide = auto()
    MidgameGoToCan = auto()
    MidgameGrabbing = auto()
    MidgameGoToZone = auto()


GREEN_ZONE = 0
RED_ZONE = 1
GOLDEN_ZONE = 2
GREEN_ZONE_OPP = 3
RED_ZONE_OPP = 4
GOLDEN_ZONE_OPP = 5

GREEN_CAN = 0
RED_CAN = 1
GOLDEN_CAN = 2


class RobotHandler():
    def __init__(self, computer_receiver: ComputerReceiver):
        self.startFrame: int = -1
        self.startTime = time.time()
        self.state = RobotState.StartScan

        # four vertices of scoring zones in world coords
        # np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        self.zones = [None, None, None, None, None, None]

        # Store planned path to cans
        self.cans: List[Tuple[float, float]] = []
        self.can_colors: List[int] = []

        # can_x, can_y, can_color
        self.current_can: Tuple[float, float, int] = (0, 0, -1)
        # zone to go to
        self.targetZone: int = -1

        # x, y, stack size, color
        self.stacked_cans = List[Tuple[float, float, int, int]]

        # waypoints to send to pi for movement
        self.waypoints: List[Tuple[float, float]] = []
        self.lastTimeSentPath = 0

        self.computer_receiver = computer_receiver
        self.isHandlingFrame = False

        self.thetaStar = ThetaStarPlanner()

        # Profiler for performance monitoring
        self.profiler = Profiler()

    def start(self):
        self.startFrame = -1
        self.startTime = time.time()

    def handleFrame(
            self,
            frame: np.ndarray,
            frame_id: int,
            robot_x: float,
            robot_y: float,
            theta: float):
        self.isHandlingFrame = True
        self.profiler.start_frame()

        result = segmentImage(frame)
        self.profiler.record("segmentImage")

        # scan and set any zones that haven't been found yet
        self.scanAndSetZones(result, frame)
        self.profiler.record("scanAndSetZones")

        # Dispatch to appropriate state handler
        if self.state == RobotState.StartScan:
            self.handleStartScan(frame_id, result, frame)
        elif self.state == RobotState.StartGather:
            self.handleStartGather(robot_x, robot_y, theta)
        elif self.state == RobotState.MidgameSearch:
            self.handleSearch()
        elif self.state == RobotState.MidgameDecide:
            self.handleMidgameDecide()
        elif self.state == RobotState.MidgameGoToCan:
            self.handleMidgameGoToCan(robot_x, robot_y, theta)
        elif self.state == RobotState.MidgameGrabbing:
            self.handleMidgameGrabbing(robot_x, robot_y, theta)
        elif self.state == RobotState.MidgameGoToZone:
            self.handleMidgameGoToZone(robot_x, robot_y, theta)
        else:
            print("ERROR: INVALID STATE")

        self.profiler.record("handleState")

        self.profiler.end_frame()
        self.isHandlingFrame = False

    # ------------------------ STATE FUNCTIONS .----------------------------

    def handleStartScan(self, frame_id: int, result, frame: np.ndarray):
        """Handle StartScan state: detect cans and plan initial path"""
        if self.startFrame == -1:
            self.startFrame = frame_id

            # ------------- PLAN PATH TO DETECTED CANS -------------
            # Get all detected cans in image coordinates
            can_locations_xy, _ = getCans(result, frame)

            # Filter cans that are on our side of the center border
            filtered_cans = [
                (x, y) for x, y in can_locations_xy
                if x < (CENTER_BORDER_X + CAN_DIAMETER)
            ]

            # Sort cans by y-coordinate (positive to negative)
            sorted_cans = sorted(filtered_cans, key=lambda p: -p[1])
            print(sorted_cans)

            # Store the planned path
            self.cans = sorted_cans
            # add green zone
            self.cans.append((4 * FT_TO_MM, 0))

            self.state = RobotState.StartGather

    def handleStartGather(self, robot_x: float, robot_y: float, theta: float):
        """Handle StartGather state: send waypoints and check if cans reached"""
        # ---------- SEND PATH IF IT HASN'T BEEN SENT YET -------------
        if time.time() - self.lastTimeSentPath > 2:
            self.lastTimeSentPath = time.time()

            self.waypoints = self.cans
            self.send_waypoints(SE2(robot_x, robot_y, theta))

        while len(self.cans) > 0 and self.isPointInScooper(
                self.cans[0], SE2(robot_x, robot_y, theta)):
            self.cans.pop(0)

    def handleSearch(self):
        """Handle Search state: rotate slowly until target zone is found"""
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

    def handleMidgameGoToCan(
            self,
            robot_x: float,
            robot_y: float,
            theta: float):
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
        if self.isPointInScooper(self.cans[0], SE2(robot_x, robot_y, theta)):
            # remove it from list cuz its gonna get moved
            self.cans.pop(0)

            # logic for grabbing it
            self.current_can = (can_x, can_y, can_color)
            self.handleMidgameGrabbing(robot_x, robot_y, theta)
        else:
            # move to can using theta*
            rx, ry = self.thetaStar.planning(robot_x, robot_y, can_x, can_y)
            self.waypoints = list(zip(rx, ry))
            self.send_waypoints(SE2(robot_x, robot_y, theta))

    def handleMidgameGrabbing(
            self,
            robot_x: float,
            robot_y: float,
            theta: float):
        """
        Grab a can that is in the scooper
        """
        self.state = RobotState.MidgameGrabbing

        cx, cy, can_color = self.current_can
        if can_color not in [GREEN_CAN, RED_CAN, GOLDEN_CAN]:
            self.state = RobotState.MidgameGoToCan
            return

        if self.isPointInGripper((cx, cy), SE2(robot_x, robot_y, theta)):
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
        elif self.isPointInScooper((cx, cy), SE2(robot_x, robot_y, theta)):
            # go closer
            self.computer_receiver.send_world_xy(cx, cy)
        else:
            self.state = RobotState.MidgameGoToCan

    def handleMidgameGoToZone(
            self,
            robot_x: float,
            robot_y: float,
            theta: float):
        """Handle MidgameGoToZone state: navigate to zone and release can"""
        # Check if robot is in target zone
        if self.zones[self.targetZone] is None:
            self.handleSearch()
            self.state = RobotState.MidgameSearch
            return

        inZone = False
        if self.zones[self.targetZone] is not None:
            inZone = isPointInPoly(
                (robot_x, robot_y), self.zones[self.targetZone])

        if inZone:
            # Release can at base height (or stack height if stacking)
            # TODO: calculate stack height based on existing cans in zone
            self.computer_receiver.send_release_can(0)
            self.state = RobotState.MidgameDecide
        else:
            # Plan path to zone center
            gx, gy = getQuadCenter(self.zones[self.targetZone])
            rx, ry = self.thetaStar.planning(robot_x, robot_y, gx, gy)
            self.waypoints = list(zip(rx, ry))
            self.send_waypoints(SE2(robot_x, robot_y, theta))

    def scanAndSetZones(self, result, image):
        """
        Detects and assigns the 6 scoring zones from YOLO results.
        Only updates zones that haven't been detected yet (are None).
        Uses x-coordinate to determine ours vs opponent: x < CENTER_BORDER_X is ours.

        Args:
            result: YOLO result object from inference
            image: Original BGR image used for zone detection

        Returns:
            bool: True if all 6 zones have been detected, False otherwise
        """
        # Get zones sorted by distance (closest first)
        quads_xy, class_names = getZones(result, image)

        if len(quads_xy) == 0:
            return all(zone is not None for zone in self.zones)

        # Iterate through all detected zones
        for quad, name in zip(quads_xy, class_names):
            # Calculate center x-coordinate to determine which side
            center_x = np.mean(quad[:, 0])
            is_our_side = center_x < CENTER_BORDER_X

            if name == 'Green Zone':
                if is_our_side and self.zones[GREEN_ZONE] is None:
                    self.zones[GREEN_ZONE] = quad
                elif not is_our_side and self.zones[GREEN_ZONE_OPP] is None:
                    self.zones[GREEN_ZONE_OPP] = quad

            elif name == 'Red Zone':
                if is_our_side and self.zones[RED_ZONE] is None:
                    self.zones[RED_ZONE] = quad
                elif not is_our_side and self.zones[RED_ZONE_OPP] is None:
                    self.zones[RED_ZONE_OPP] = quad

            elif name == 'Golden Zone':
                if is_our_side and self.zones[GOLDEN_ZONE] is None:
                    self.zones[GOLDEN_ZONE] = quad
                elif not is_our_side and self.zones[GOLDEN_ZONE_OPP] is None:
                    self.zones[GOLDEN_ZONE_OPP] = quad

        # Check if all zones have been detected
        all_detected = all(zone is not None for zone in self.zones)
        return all_detected

    # ------------------------ HELPER FUNCTIONS .----------------------------

    def isPointInScooper(
        self,
        point: Tuple[float, float],
        robot_pose: SE2
    ) -> bool:
        """
        Check if a point is within reach of the robot (circular radius OR forward rectangle).

        Uses SE(2) transformation to convert to robot-relative coordinates, then checks:
        1. Within circular radius: (BASE_D - CAN_DIAMETER) / 2
        2. Within rectangle: SCOOPER_LENGTH forward, (BASE_D - CAN_DIAMETER) wide

        Args:
            point: (x, y) world coordinates in mm
            robot_pos: (x, y) robot position in mm
            robot_heading: Robot heading in radians

        Returns:
            bool: True if point is reachable
        """

        # Transform point to robot's local coordinate frame using SE2
        local_x, local_y = world_to_relative(point, robot_pose)

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

    def isPointInGripper(
        self,
        point: Tuple[float, float],
        robot_pose: SE2
    ) -> bool:
        """
        Check if a point is within reach of the robot (circular radius OR forward rectangle).

        Uses SE(2) transformation to convert to robot-relative coordinates, then checks:
        1. Within circular radius: (BASE_D - CAN_DIAMETER) / 2
        2. Within rectangle: SCOOPER_LENGTH forward, (BASE_D - CAN_DIAMETER) wide

        Args:
            point: (x, y) world coordinates in mm
            robot_pos: (x, y) robot position in mm
            robot_heading: Robot heading in radians

        Returns:
            bool: True if point is reachable
        """
        local_x, local_y = world_to_relative(point, robot_pose)

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

    def send_waypoints(self, robot_pose: SE2):
        plan = get_movement_plan(self.waypoints, robot_pose)

        movement_args = []
        for move in plan:
            dist, theta = move
            movement_args.extend(get_rotate(theta))
            movement_args.extend(get_forward_mm(dist))

        self.computer_receiver.override_movement(movement_args)
        self.lastTimeSentPath = time.time()

import time
import math
import numpy as np
from enum import Enum
from typing import Tuple, List
from spatialmath import SE2
from connection.ComputerReceiver import ComputerReceiver
from nav import get_forward_mm, get_rotate
from yolo.segment import segmentImage
from yolo.zone_utils import getZones
from yolo.can_utils import getCans
from config import CENTER_BORDER_X, CAN_DIAMETER, BASE_D, FT_TO_MM, SCOOPER_LENGTH
from coordinates.relativeCoordinates import get_movement_plan, world_to_relative
from profiler import Profiler


class RobotState(Enum):
    StartScan = 1
    StartGather = 2
    MoveToZone = 3


GREEN_ZONE = 0
RED_ZONE = 1
GOLDEN_ZONE = 2
GREEN_ZONE_OPP = 3
RED_ZONE_OPP = 4
GOLDEN_ZONE_OPP = 5


class RobotHandler():
    def __init__(self, computer_receiver: ComputerReceiver):
        self.startFrame: int = -1
        self.startTime = time.time()
        self.state = RobotState.StartScan

        # four vertices of scoring zones in world coords
        # np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        self.zones = [None, None, None, None, None, None]

        # Store planned path to cans
        self.planned_path: List[Tuple[float, float]] = []
        self.lastTimeSentPath = 0

        self.computer_receiver = computer_receiver
        self.isHandlingFrame = False

        # Profiler for performance monitoring
        self.profiler = Profiler()

    def start(self):
        self.startFrame = -1
        self.startTime = time.time()

    def handleFrame(
            self,
            frame: np.ndarray,
            frame_id: int,
            x: float,
            y: float,
            theta: float):
        self.profiler.start()
        self.isHandlingFrame = True

        result = segmentImage(frame)
        self.profiler.record("segmentImage")

        getZones(result, frame)
        self.profiler.record("getZones")

        if self.state == RobotState.StartScan:
            if self.startFrame == -1:
                self.startFrame = frame_id

                # ------------- PLAN PATH TO DETECTED CANS -------------
                # Get all detected cans in image coordinates
                can_locations_xy, _ = getCans(result, frame)
                self.profiler.record("getCans")

                # Filter cans that are on our side of the center border
                filtered_cans = [
                    (y * 8, x * 8) for x, y in can_locations_xy
                    if x < (CENTER_BORDER_X + CAN_DIAMETER)
                ]

                # Sort cans by y-coordinate (positive to negative)
                sorted_cans = sorted(filtered_cans, key=lambda p: -p[1])
                print(sorted_cans)

                # Store the planned path
                self.planned_path = sorted_cans
                # add green zone
                self.planned_path.append((4 * FT_TO_MM, 0))
                self.state = RobotState.StartGather
                self.profiler.record("path_planning")

        elif self.state == RobotState.StartGather:
            # ---------- SEND PATH IF IT HASN'T BEEN SENT YET -------------
            if time.time() - self.lastTimeSentPath > 2:
                self.lastTimeSentPath = time.time()

                plan = get_movement_plan(self.planned_path, SE2(x, y, theta))

                movement_args = []
                for move in plan:
                    dist, theta = move
                    movement_args.extend(get_rotate(theta))
                    movement_args.extend(get_forward_mm(dist))

                self.computer_receiver.override_movement(movement_args)
                self.profiler.record("send_movement")

            while len(self.planned_path) > 0 and self.is_point_in_reach(
                    self.planned_path[0], (x, y), theta):
                self.planned_path.pop(0)
            self.profiler.record("check_reached")

        elif self.state == RobotState.MoveToZone:
            pass
        else:
            print("ERROR: INVALID STATE")

        self.isHandlingFrame = False
        self.profiler.end_frame()

    def getOurZones(self, result, image):
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

    def is_point_in_reach(
        self,
        point: Tuple[float, float],
        robot_pos: Tuple[float, float],
        robot_heading: float,
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
        robot_x, robot_y = robot_pos

        # Check 2: Is point within rectangle in front of robot?
        # Transform point to robot's local coordinate frame using SE2
        robot_pose = SE2(robot_x, robot_y, robot_heading)
        local_x, local_y = world_to_relative(point, robot_pose)

        # Check 1: Is point within circular radius?
        distance = math.sqrt(local_x**2 + local_y**2)
        if distance <= (BASE_D - CAN_DIAMETER) / 2:
            return True

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

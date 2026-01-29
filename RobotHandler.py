import time
import math
import numpy as np
from enum import Enum, auto
from typing import Tuple, List, Optional
from spatialmath import SE2
from IRobotCommander import IRobotCommander  # type: ignore
from connection.frame_info import FrameInfo
from navHelpers import get_rotate
from vision.pixelTo3D import is_world_point_visible
from vision.segment import segmentImage
from vision.zone_utils import doPolygonsIntersect, getPolygonCenter, getZones
from vision.can_utils import getCans, is_hull_overlap_with_target_rect
from vision.relativeCoordinates import relative_to_world, world_to_relative
from vision.mask_utils import maskToConvexHull, yoloMaskToBinary
from profiler import Profiler
from thetaStar import ThetaStar
from streamer import Streamer
from config import FPS, CAN_DIAMETER, BASE_D, CLAW_OFFSET, PICKED_RECT, ROBOT_DIAMETER, SCOOPER_LENGTH, TEMP_STACK_OFFSET
from colors import GREEN_CAN, GREEN_ZONE, GREEN_ZONE_OPP, RED_CAN, RED_ZONE, RED_ZONE_OPP, GOLDEN_CAN, GOLDEN_ZONE, GOLDEN_ZONE_OPP, ZONE_CLASS_NAMES, canNamesToNumbers


class RobotState(Enum):
    StartScan = auto()
    StartGather = auto()
    SearchForZone = auto()
    SearchForCan = auto()
    MidgameGoToCan = auto()
    MidgameGrabbing = auto()
    PlaceInZone = auto()
    PickupStack = auto()
    AddStack = auto()
    FinishedStacking = auto()
    PostGrab = auto()


class RobotHandler():
    def __init__(self, robot_commander: IRobotCommander):
        # state variables
        self.state = RobotState.StartScan
        self.started = False
        self.paused = False
        # Command ID we're waiting for (0 = not waiting)
        self.waiting_for_command_id = 0

        # BEST GUESS MEMORY VARIABLES
        # Polygon vertices of scoring zones in world coords (mm)
        # list of zones, each zone is [[x1, y1], [x2, y2], ..., [xN, yN]]
        self.zones: List[Optional[np.ndarray]] = [
            None, None, None, None, None, None]
        self.zone_confidences = [0, 0, 0, 0, 0, 0]
        # Store planned path to cans
        self.cans: List[Tuple[float, float]] = []
        self.can_colors: List[int] = []
        # Number of consecutive frames each can has been visible but not detected
        # Aligned by index with self.cans / self.can_colors
        self.can_miss_counts: List[int] = []
        # Can detection confidence tracking: rounded (x,y) -> consecutive
        # frames detected
        self.can_detections: dict[Tuple[int, int], int] = {}
        self.DETECTION_THRESHOLD = 3  # Require N consecutive frames to confirm
        # x, y, stack size, color
        self.stacked_cans: List[Tuple[float, float, int, int]] = []
        self.MAX_STACK_SIZE = 2
        self.borders: List[Tuple[int, int]] = []

        # VARIABLES FOR CURRENT STATE
        # can_x, can_y, can_color
        self.current_can: Tuple[float, float, int] = (0, 0, -1)
        # zone to go to
        self.targetZone: int = -1
        # target of stacking state: stack id
        self.targetStackId: int = -1
        self.newStackPosition: Optional[Tuple[float, float]] = None

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

        # Skip processing if paused
        if self.paused:
            # if self.frame_id % 30 == 0:  # Print every 30 frames (~1 second)
            #     print("⏸️  PAUSED (type 'resume' to continue)")
            self.profiler.end_frame()
            return

        # Check if we're waiting for a command to complete
        if self.waiting_for_command_id > 0:
            # Check if the command we're waiting for has completed
            if frame_info.lastCompletedCommandId >= self.waiting_for_command_id:
                # Command completed, clear waiting state
                print(
                    f"✓ Command {self.waiting_for_command_id} completed (frame {self.frame_id})")
                self.waiting_for_command_id = 0
            else:
                # Still waiting, skip state processing
                self.profiler.end_frame()
                return

        # Use top camera frame for vision processing
        self.frame_top = frame_info.frame_top
        self.frame_bottom = frame_info.frame_bottom
        self.frame_id = frame_info.frame_id
        self.robot_pose = SE2(frame_info.x, frame_info.y, frame_info.theta)
        self.distanceSensed = frame_info.distanceSensed

        self.result_top = segmentImage(self.frame_top)
        self.result_bottom = segmentImage(self.frame_bottom)
        self.profiler.record("segmentImage")

        for result, frame, is_top in [
            (self.result_top, self.frame_top, True),
            (self.result_bottom, self.frame_bottom, False)
        ]:
            self.scanAndSetZones(result, frame, is_top)

        self.updateCanDetections()

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
        elif self.state == RobotState.SearchForZone:
            self.handleSearchForZone()
        elif self.state == RobotState.SearchForCan:
            self.handleSearchForCan()
        elif self.state == RobotState.MidgameGoToCan:
            self.handleMidgameGoToCan()
        elif self.state == RobotState.MidgameGrabbing:
            self.handleMidgameGrabbing()
        elif self.state == RobotState.PlaceInZone:
            self.handlePlaceInZone()
        elif self.state == RobotState.PickupStack:
            self.handlePickUpStack()
        elif self.state == RobotState.AddStack:
            self.handleAddStack()
        elif self.state == RobotState.FinishedStacking:
            self.handleFinishedStacking()
        elif self.state == RobotState.PostGrab:
            self.handlePostGrab()

        self.profiler.record("handleState")

        self.updateTelemetry()
        self.profiler.record("telemetry")

        # self.paused = True
        # time.sleep(5)
        self.profiler.record("sleep")

        self.profiler.end_frame()

    def updateCanDetections(self) -> None:
        """
        Detect cans from both camera segmentations, then update:
        - self.cans / self.can_colors
        - self.can_detections (stability tracking)
        - self.can_miss_counts (persistence when temporarily missed)
        """
        # Collect all detections from both cameras
        all_locations: List[Tuple[float, float]] = []
        all_colors: List[int] = []

        for result, frame, is_top in [
            (self.result_top, self.frame_top, True),
            (self.result_bottom, self.frame_bottom, False)
        ]:
            locations, color_strings = getCans(result, frame, is_top)
            print(locations)
            colors = canNamesToNumbers(color_strings)

            # Transform to world coordinates
            locations = [relative_to_world(location, self.robot_pose)
                         for location in locations]

            all_locations.extend(locations)
            all_colors.extend(colors)

        # Update detection confidence tracking
        new_detections: dict[Tuple[int, int], int] = {}

        for i, location in enumerate(all_locations):
            # Round to 50mm grid for matching
            rounded = (
                round(
                    location[0] /
                    50) *
                50,
                round(
                    location[1] /
                    50) *
                50)

            # Check if matches existing detection
            matched = False
            for existing_loc, count in self.can_detections.items():
                if getDistance(rounded, existing_loc) < CAN_DIAMETER:
                    new_detections[existing_loc] = count + 1
                    matched = True
                    break

            if not matched:
                new_detections[rounded] = 1

        # Build confirmed cans list (detections >= threshold)
        confirmed_cans: List[Tuple[float, float]] = []
        confirmed_colors: List[int] = []
        confirmed_miss_counts: List[int] = []

        for rounded_loc, count in new_detections.items():
            if count >= self.DETECTION_THRESHOLD:
                # Find the actual detection location (not rounded)
                for i, location in enumerate(all_locations):
                    rounded = (
                        round(
                            location[0] /
                            50) *
                        50,
                        round(
                            location[1] /
                            50) *
                        50)
                    if rounded == rounded_loc:
                        confirmed_cans.append(location)
                        confirmed_colors.append(all_colors[i])
                        confirmed_miss_counts.append(0)
                        break

        # Keep old cans with miss count tracking
        MAX_MISS_COUNT = 10  # Remove can after 10 consecutive frames without detection

        for i in range(len(self.cans)):
            old_can = self.cans[i]
            old_color = self.can_colors[i]
            old_miss_count = self.can_miss_counts[i] if i < len(
                self.can_miss_counts) else 0

            # Skip if already in confirmed list (was detected this frame)
            if any(getDistance(old_can, new_can) < CAN_DIAMETER / 2
                   for new_can in confirmed_cans):
                continue

            # Check visibility in both cameras
            visible_top = is_world_point_visible(
                old_can[0], old_can[1], self.robot_pose, True)
            visible_bottom = is_world_point_visible(
                old_can[0], old_can[1], self.robot_pose, False)

            # If not visible in either camera, keep with same miss count (out
            # of view)
            if not visible_top and not visible_bottom:
                confirmed_cans.append(old_can)
                confirmed_colors.append(old_color)
                confirmed_miss_counts.append(old_miss_count)
            # If visible but not detected, increment miss count
            elif old_miss_count < MAX_MISS_COUNT:
                confirmed_cans.append(old_can)
                confirmed_colors.append(old_color)
                confirmed_miss_counts.append(old_miss_count + 1)
            # else: visible but missed too many times, remove it

        self.can_detections = new_detections
        self.cans = confirmed_cans
        self.can_colors = confirmed_colors
        self.can_miss_counts = confirmed_miss_counts

    # ------------------------ STATE FUNCTIONS .----------------------------

    def handleStartScan(self, frame_id: int):
        """Handle StartScan state: detect cans and plan initial path"""
        self.state = RobotState.StartScan
        if self.startFrame == -1:
            self.startFrame = frame_id
            # print("→ reset_gripper")
            # self.robot_commander.reset_gripper()

        if self.started:
            print(f"State: {self.state.name} → StartGather")
            self.state = RobotState.StartGather

    def handleStartGather(self):
        """Handle StartGather state: send waypoints and check if cans reached"""
        self.state = RobotState.StartGather

        self.targetZone = GREEN_ZONE
        print(f"State: {self.state.name} → MidgameGoToCan")
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

    def handleSearchForZone(self):
        """Handle SearchForZone state: rotate slowly until target zone is found"""
        self.state = RobotState.SearchForZone
        # Check if target zone has been found
        if self.zones[self.targetZone] is None:
            # Rotate slowly (45 degrees every second)
            rotate_cmd = list(get_rotate(math.pi / 4 / FPS))
            print("→ override_movement(rotate)")
            self.robot_commander.override_movement(rotate_cmd)
        else:
            print(f"State: {self.state.name} → PlaceInZone (zone found)")
            self.state = RobotState.PlaceInZone

    def handleSearchForCan(self):
        """
        Spin in place until we see at least one can, then go to the nearest can.
        """
        self.state = RobotState.SearchForCan
        if len(self.cans) == 0:
            # Rotate slowly while searching
            rotate_cmd = list(get_rotate(math.pi / 2 / FPS))
            print("→ override_movement(search_for_can_rotate)")
            self.robot_commander.override_movement(rotate_cmd)
            return

        print(f"State: {self.state.name} → MidgameGoToCan (cans found)")
        self.handleMidgameGoToCan()

    def handleMidgameGoToCan(self):
        """
        Handle MidgameGoToCan state: navigate to a can
        """
        self.state = RobotState.MidgameGoToCan
        if len(self.cans) == 0:
            print(f"State: {self.state.name} → SearchForCan (no cans left)")
            self.state = RobotState.SearchForCan
            self.handleSearchForCan()
            return

        # Filter out cans that are effectively already part of / too close to a known stack
        # (prevents repeatedly targeting stacked cans as "loose" cans)
        if len(self.stacked_cans) > 0:
            filtered_cans: List[Tuple[float, float]] = []
            filtered_colors: List[int] = []

            for can, color in zip(self.cans, self.can_colors):
                too_close_to_stack = False
                for stack_x, stack_y, stack_size, _ in self.stacked_cans:
                    # Only consider stacks that actually exist (size > 0)
                    if stack_size <= 0:
                        continue
                    if getDistance(can, (stack_x, stack_y)) <= CAN_DIAMETER:
                        too_close_to_stack = True
                        break

                if not too_close_to_stack:
                    filtered_cans.append(can)
                    filtered_colors.append(color)

            self.cans = filtered_cans
            self.can_colors = filtered_colors

            if len(self.cans) == 0:
                print(
                    f"State: {self.state.name} → SearchForCan (only stacked cans visible)")
                self.state = RobotState.SearchForCan
                self.handleSearchForCan()
                return

        # Sort cans by distance from robot (nearest first)
        robot_x, robot_y = self.robot_pose.x, self.robot_pose.y
        robot_pos = (robot_x, robot_y)
        sorted_pairs = sorted(
            zip(self.cans, self.can_colors),
            key=lambda pair: getDistance(robot_pos, pair[0])
        )
        self.cans = [can for can, _ in sorted_pairs]
        self.can_colors = [color for _, color in sorted_pairs]

        # this allows for dynamic update of which can to go to
        can_x, can_y = self.cans[0]
        can_color = self.can_colors[0]
        if self.isPointClose(can_x, can_y):
            # remove it from list cuz its gonna get moved
            self.cans.pop(0)

            # logic for grabbing it
            self.current_can = (can_x, can_y, can_color)
            print(
                f"State: {self.state.name} → MidgameGrabbing (can at {can_x:.0f}, {can_y:.0f})")
            self.handleMidgameGrabbing()
        else:
            # move to can using theta*
            self.thetaStarAndSend(can_x, can_y)

    def handleMidgameGrabbing(self):
        """
        Grab a can that is in the scooper
        """
        self.state = RobotState.MidgameGrabbing

        cx, cy, _ = self.current_can
        if self.isPointInGripper(cx, cy):
            # figure out if tipped and pick up
            if self.hasTippedCan():
                print("→ approach_can_with_ds, pickup_tipped_can, release_can")
                self.robot_commander.approach_can_with_ds()
                self.robot_commander.pickup_tipped_can()
                self.robot_commander.release_can()
                self.waiting_for_command_id = self.robot_commander.get_last_command_id()
                print(
                    f"State: {self.state.name} → MidgameGoToCan (tipped can released)")
                self.state = RobotState.MidgameGoToCan
            else:
                print("→ approach_can_with_ds, pickup_can")
                self.robot_commander.approach_can_with_ds()
                self.robot_commander.waitFinishedMoving()
                self.robot_commander.pickup_can()
                self.waiting_for_command_id = self.robot_commander.get_last_command_id()
                print(f"State: {self.state.name} → PostGrab")
                self.state = RobotState.PostGrab
        elif self.isPointClose(cx, cy):
            print("→ approach_can_with_ds, pickup_can")
            # get close then pick up
            self.robot_commander.approach_can_with_ds()
            self.robot_commander.pickup_can()
            self.waiting_for_command_id = self.robot_commander.get_last_command_id()
            print(f"State: {self.state.name} → PostGrab")
            self.state = RobotState.PostGrab
        else:
            self.thetaStarAndSend(cx, cy)
            print(
                f"State: {self.state.name} → MidgameGoToCan (navigating to can)")
            self.state = RobotState.MidgameGoToCan

    def handlePostGrab(self):
        self.state = RobotState.PostGrab

        can_color = self.current_can[2]
        if can_color not in [GREEN_CAN, RED_CAN, GOLDEN_CAN]:
            print(
                f"State: {self.state.name} → MidgameGoToCan (invalid can color)")
            self.state = RobotState.MidgameGoToCan
            return

        if True:
            # if self.hasGoodPickup():
            # Select target zone based on can color
            # if can_color == GREEN_CAN:
            #     self.targetZone = GREEN_ZONE
            #     zone_name = "GREEN"
            # elif can_color == RED_CAN:
            #     self.targetZone = RED_ZONE
            #     zone_name = "RED"
            # else:  # GOLDEN_CAN
            #     self.targetZone = GOLDEN_ZONE
            #     zone_name = "GOLDEN"
            self.targetZone = GREEN_ZONE
            zone_name = "GREEN"
            print(
                f"State: {self.state.name} → PlaceInZone (target: {zone_name})")
            self.targetStackId = -1
            self.handlePlaceInZone()
        else:
            print("→ release_can")
            print(f"State: {self.state.name} → MidgameGoToCan (bad pickup)")
            self.robot_commander.release_can()
            self.current_can = (0, 0, -1)
            self.handleMidgameGoToCan()

    def handlePlaceInZone(self):
        """
            Once the robot is already holding a can
        """
        self.state = RobotState.PlaceInZone

        if self.newStackPosition is None:
            # pick some point that is good enough distance away
            zone_x, zone_y = getPolygonCenter(self.zones[self.targetZone])

            # use self.targetStackId to remember which stack
            # remember to set to -1 when starting stacking phase
            targetStack = None
            if self.targetStackId == -1:
                # find a stack based on the color zone
                for i, stack in enumerate(self.stacked_cans):
                    x, y, size, color = stack
                    if size < self.MAX_STACK_SIZE and color == self.targetZone:
                        # choose this stack
                        self.targetStackId = i
                        targetStack = stack
                        break
                if targetStack is None:
                    # make new stack
                    targetStack = (zone_x, zone_y, 0, self.targetZone)
                    self.stacked_cans.append(targetStack)
                    self.targetStackId = len(self.stacked_cans) - 1
            else:
                targetStack = self.stacked_cans[self.targetStackId]

            # get the target
            cx, cy, size, color = targetStack

            # calculate offsetted position
            dx = zone_x - cx
            dy = zone_y - cy
            distance = math.sqrt(dx * dx + dy * dy)

            # Move 200mm towards zone center (or less if zone center is closer)
            if distance > 0.01:
                new_pos = (cx + dx / distance * TEMP_STACK_OFFSET,
                           cy + dy / distance * TEMP_STACK_OFFSET)
            else:
                # Stack is already at zone center, offset in x direction
                new_pos = (cx + TEMP_STACK_OFFSET, cy)
            self.newStackPosition = new_pos

        gx, gy = self.newStackPosition
        if self.isPointInGripper(gx, gy):
            self.robot_commander.set_down_can()
            self.robot_commander.backup()
            self.robot_commander.waitFinishedMoving()
            self.waiting_for_command_id = self.robot_commander.get_last_command_id()
            if self.stacked_cans[self.targetStackId][2] == 0:
                self.state = RobotState.MidgameGoToCan
            else:
                self.state = RobotState.PickupStack
        else:
            self.thetaStarAndSend(gx, gy)

    def handlePickUpStack(self):
        self.state = RobotState.PickupStack

        targetStack = None
        if self.targetStackId == -1:
            # find a stack based on the color zone
            self.targetStackId = -1
            for i, stack in enumerate(self.stacked_cans):
                x, y, size, color = stack
                if size < self.MAX_STACK_SIZE and color == self.targetZone:
                    # choose this stack
                    self.targetStackId = i
                    targetStack = stack
                    break
            if targetStack is None:
                # make new stack
                zone_x, zone_y = getPolygonCenter(self.zones[self.targetZone])
                targetStack = (zone_x, zone_y, 0, self.targetZone)
                self.stacked_cans.append(targetStack)
                self.targetStackId = len(self.stacked_cans) - 1
        else:
            targetStack = self.stacked_cans[self.targetStackId]

        gx, gy, size, color = targetStack
        if self.isPointInGripper(gx, gy):
            self.robot_commander.approach_can_with_ds()
            self.robot_commander.pickup_can()
            self.waiting_for_command_id = self.robot_commander.get_last_command_id()
            self.state = RobotState.AddStack
        else:
            self.thetaStarAndSend(gx, gy)

    def handleAddStack(self):
        """
            Assuming holding a stack right now
        """
        self.state = RobotState.AddStack

        assert (self.newStackPosition is not None)

        gx, gy = self.newStackPosition
        if self.isPointClose(gx, gy):
            self.robot_commander.approach_can_with_ds()
            self.robot_commander.release_can()
            self.state = RobotState.FinishedStacking
        else:
            self.thetaStarAndSend(gx, gy)

    def handleFinishedStacking(self):
        self.state = RobotState.FinishedStacking
        # get position of stacked cans, which should be right in front after
        # stacking
        cx, cy = relative_to_world(
            (CLAW_OFFSET + CAN_DIAMETER / 2, 0), self.robot_pose)

        # update list of stacked cans
        updated = False
        for i in range(len(self.stacked_cans)):
            _, _, prev_size, color = self.stacked_cans[i]
            if self.targetStackId == i:
                self.stacked_cans[i] = (cx, cy, prev_size + 1, color)
                updated = True
                break
        if not updated:
            print("WARNING: NO STACK FOUND TO UPDATE AFTER FINISH STACK")

        self.thetaStar.addCan(cx, cy)

        print("→ override_movement([-1, -1, 3000])")
        self.robot_commander.backup()
        self.waiting_for_command_id = self.robot_commander.get_last_command_id()
        print(f"⏳ Waiting for command {self.waiting_for_command_id}")
        print(f"State: {self.state.name} → MidgameGoToCan")
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
        squares_xy, class_names, confidences = getZones(result, image, is_top)

        # Iterate through all detected zones
        for zone, name, conf in zip(squares_xy, class_names, confidences):
            if name == ZONE_CLASS_NAMES[GREEN_ZONE]:
                self.updateZone(zone, conf, GREEN_ZONE, GREEN_ZONE_OPP)

            elif name == ZONE_CLASS_NAMES[RED_ZONE]:
                self.updateZone(zone, conf, RED_ZONE, RED_ZONE_OPP)

            elif name == ZONE_CLASS_NAMES[GOLDEN_ZONE]:
                self.updateZone(zone, conf, GOLDEN_ZONE, GOLDEN_ZONE_OPP)

    def hasGoodPickup(
        self,
        min_intersection_over_rect: float = 0.2,
        max_segmentation_over_rect: float = 15.6,
    ) -> bool:
        """
        Check whether any current segmentation mask overlaps well with the target rectangle.

        For each mask in the given YOLO result, this function:
        - converts the mask to a convex hull in pixel coordinates
        - checks the hull's area
        - computes its overlap with the fixed target rectangle in image space
          using ``is_hull_overlap_with_target_rect``

        It returns True as soon as any hull satisfies:
        - intersection_area / rect_area >= min_intersection_over_rect
        - hull_area / rect_area <= max_segmentation_over_rect

        Args:
            min_intersection_over_rect: Minimum required fraction of the
                rectangle's area that must be covered by the intersection
                (0.0–1.0).
            max_segmentation_over_rect: Maximum allowed fraction of the
                rectangle's area that may be covered by the hull itself
                (0.0–1.0).

        Returns:
            bool: True if any segmentation's convex hull overlaps the rectangle
            sufficiently and is under the area threshold, False otherwise.
        """
        if self.result_bottom is None or self.result_bottom.masks is None:
            return False

        # Choose corresponding image for mask resizing; here we assume bottom
        # camera result when this helper is used from handleFrame.
        image = self.frame_bottom

        print(len(self.result_bottom.masks))
        for mask_orig in self.result_bottom.masks:
            try:
                # Convert YOLO mask object to a binary numpy mask in image
                # space
                binary_mask = yoloMaskToBinary(mask_orig, image)
                hull_uv = maskToConvexHull(binary_mask)
                if hull_uv is None or len(hull_uv) == 0:
                    continue
            except Exception:
                continue

            if is_hull_overlap_with_target_rect(
                hull_uv,
                PICKED_RECT,
                min_intersection_over_rect,
                max_segmentation_over_rect,
            ):
                return True

        return False

    def hasTippedCan(
        self,
        min_intersection_over_rect: float = 0.7,
        max_segmentation_over_rect: float = 1.6,
    ) -> bool:
        """
        Check whether any current segmentation mask overlaps well with the target rectangle.

        For each mask in the given YOLO result, this function:
        - converts the mask to a convex hull in pixel coordinates
        - checks the hull's area
        - computes its overlap with the fixed target rectangle in image space
          using ``is_hull_overlap_with_target_rect``

        It returns True as soon as any hull satisfies:
        - intersection_area / rect_area >= min_intersection_over_rect
        - hull_area / rect_area <= max_segmentation_over_rect

        Args:
            min_intersection_over_rect: Minimum required fraction of the
                rectangle's area that must be covered by the intersection
                (0.0–1.0).
            max_segmentation_over_rect: Maximum allowed fraction of the
                rectangle's area that may be covered by the hull itself
                (0.0–1.0).

        Returns:
            bool: True if any segmentation's convex hull overlaps the rectangle
            sufficiently and is under the area threshold, False otherwise.
        """
        if self.result_bottom is None or self.result_bottom.masks is None:
            return False

        # Choose corresponding image for mask resizing; here we assume bottom
        # camera result when this helper is used from handleFrame.
        image = self.frame_bottom

        for mask_orig in self.result_bottom.masks:
            try:
                # Convert YOLO mask object to a binary numpy mask in image
                # space
                binary_mask = yoloMaskToBinary(mask_orig, image)
                hull_uv = maskToConvexHull(binary_mask)
                if hull_uv is None or len(hull_uv) == 0:
                    continue
            except Exception:
                continue

            if is_hull_overlap_with_target_rect(
                hull_uv,
                PICKED_RECT,
                min_intersection_over_rect,
                max_segmentation_over_rect,
            ):
                return True

        return False

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
                prev_x, prev_y = getPolygonCenter(prev_zone)
                prevDistSquared = prev_x * prev_x + prev_y * prev_y
                curr_x, curr_y = getPolygonCenter(zone)
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

    def send_waypoints_with_start(self, waypoints: List[Tuple[float, float]]):
        x, y, _ = unpackPose(self.robot_pose)
        command_args = [x, y]
        for x, y in waypoints:
            command_args.append(x)
            command_args.append(y)
        self.robot_commander.override_waypoints(command_args)

    def thetaStarAndSend(self, x: float, y: float):
        # temporary while thetastar doesn't work
        # dx, dy = world_to_relative((x, y), self.robot_pose)
        # gx, gy = relative_to_world((max(dx - 80, 0), dy), self.robot_pose)
        # self.robot_commander.override_world_xy(gx, gy)
        # print(f"→ override_world_xy({x:.0f}, {y:.0f})")
        # self.robot_commander.override_world_xy(x, y)
        gx, gy = x, y
        robot_x, robot_y, theta = unpackPose(self.robot_pose)
        self.thetaStar.set_start(robot_x, robot_y)
        self.thetaStar.set_goal(gx, gy)
        waypoints = self.thetaStar.path_find()
        self.send_waypoints_with_start(waypoints)

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
            # Convert dicts with non-JSON keys (e.g., tuple keys) into
            # JSON-safe dicts
            elif isinstance(v, dict):
                safe_dict = {}
                for dk, dv in v.items():
                    if isinstance(dk, tuple):
                        safe_key = ",".join(str(x) for x in dk)
                    else:
                        safe_key = dk
                    safe_dict[safe_key] = dv
                result[k] = safe_dict
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

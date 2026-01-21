import time
import numpy as np
from enum import Enum
from yolo.segment import segmentImage
from yolo.zone_utils import getZones
from config import CENTER_BORDER_X


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
    def __init__(self):
        self.startFrame: int = -1
        self.startTime = time.time()
        self.state = RobotState.StartScan

        # four vertices of scoring zones in world coords
        # np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        self.zones = [None, None, None, None, None, None]

    def start(self):
        self.startFrame = -1
        self.startTime = time.time()

    def handleFrame(self, frame: np.ndarray, frame_id: int):
        result = segmentImage(frame)
        getZones(result, frame)

        if self.state == RobotState.StartScan:
            if self.startFrame == -1:
                self.startFrame = frame_id

        elif self.state == RobotState.StartGather:
            pass
        elif self.state == RobotState.MoveToZone:
            pass
        else:
            print("ERROR: INVALID STATE")

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

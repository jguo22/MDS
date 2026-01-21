import time
import numpy as np
from enum import Enum


class RobotState(Enum):
    StartScan = 1
    StartGather = 2
    MoveToZone = 3


class RobotHandler():
    def __init__(self):
        self.startFrame: int = -1
        self.startTime = time.time()
        self.state = RobotState.StartScan

        # four vertices of scoring zones in world coords
        # np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
        self.greenZone = None
        self.redZone = None
        self.yellowZone = None
        self.greenZoneOpp = None
        self.redZoneOpp = None
        self.yellowZoneOpp = None

    def start(self):
        self.startFrame = -1
        self.startTime = time.time()

    def handleFrame(self, frame: np.ndarray, frame_id: int):
        if self.state == RobotState.StartScan:
            if self.startFrame == -1:
                self.startFrame = frame_id

        elif self.state == RobotState.StartGather:
            pass
        elif self.state == RobotState.MoveToZone:
            pass
        else:
            print("ERROR: INVALID STATE")

    def getOurZones(self):
        pass

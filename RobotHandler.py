import time
import numpy as np


class RobotHandler():
    def __init__(self):
        self.startFrame: int = -1
        self.startTime = time.time()

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
        if self.startFrame == -1:
            self.startFrame = frame_id

        if self.startFrame == frame_id:
            #
            pass

        pass

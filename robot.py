import time
import cv2

from enum import Enum
from ultralytics import YOLO

from test import RavenMotorControllers
from nav import Nav
from nav import NavMove

class RobotState(Enum):
    SEARCHING = 1
    CHECKING_SEARCH = 2
    SEEKING_CORRECTION = 3
    PICKING_UP = 4
    RETURNING = 5
    DROPPING_OFF = 6
    SEEKING_MOVING = 7
    REFINDING_OBJECT = 8
    STEP_1 = 9
    STEP_2 = 10
    STEP_3 = 11
    STEP_4 = 12
    STEP_5 = 13

MIDPOINT = 320
MARGIN = 40

class Robot:
    def __init__(self, bno):
        self.state = RobotState.CHECKING_SEARCH
        self.state_start = time.monotonic()
        self.now = time.monotonic()

        # Initialize model
        print("initializing yolo model from robot.py")
        self.model = YOLO("yolo/last_ncnn_model", task='detect')
        self.labels = self.model.names

        self.motors = RavenMotorControllers()

        self.nav = Nav(bno)
        self.cap = cv2.VideoCapture(0)

    def setNowTime(self):
        self.now = time.monotonic()

    # Move towards a human
    def moveToHuman(self):
        self.state = RobotState.SEEKING_MOVING
        self.state_start = self.now
        # Point towards human
        self.motors.stopRotating()
        self.motors.setSpeed(20)
        self.motors.setTorque(20)
    # Spin around to look for objects
    def searchMode(self):
        self.state = RobotState.SEARCHING
        self.state_start = self.now
        self.motors.rotateInPlace(40)
    # Code to obtain the golden pringle can
    # Only run this at the very start, when the robot is at the start zone
    # Returns if the robot found a golden pringle can and is going to it
    def fetchGoldenPringleCan(self):
        detections = self.getDetections()
        golden_pringle_x = 0
        golden_pringle_conf = 0

        # Go through each detection and get bbox coords, confidence, and class
        for detection in detections:

            # Get bounding box coordinates
            # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
            xyxy_tensor = detection.xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

            # Get bounding box class ID and name
            classidx = int(detection.cls.item())
            classname = self.labels[classidx]

            # Get bounding box confidence
            conf = detection.conf.item()
            print(f'found {classname}')

            # Only get confident golden pringle cans
            if conf < 0.7 or conf < golden_pringle_conf or classname != "Golden Can":
                continue

            golden_pringle_conf = conf
            golden_pringle_x = (xmax + xmin) / 2

        if (golden_pringle_conf != 0):
            print(f'found golden pringle can at x: {golden_pringle_x}')
            self.nav.move_to(1000, 1270)
            #self.nav.addPath(NavMove(1.0, 1.0, self.nav.mm_to_ticks(608), False))
            return True
        return False
    def getDetections(self):
        if not self.cap or not self.cap.isOpened():
            print(f'ERROR: CAMERA ISNT WORKING')
            return []
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print(f'ERROR: NO FRAME RETURNED')
            return []
        # Run inference on frame
        results = self.model(frame, verbose=False)
        # Extract results
        detections = results[0].boxes
        return detections
    # Returns if there's a pringle can in view
    def seesPringleCan(self):

        ret, frame = cap.read()
        # Run inference on frame
        results = self.model(frame, verbose=False)
        # Extract results
        detections = results[0].boxes

        # Variables for detected humans
        humans_detected = False
        x_mid = 320
        biggest_human_area = 0

        # Go through each detection and get bbox coords, confidence, and class
        for detection in detections:

            # Get bounding box coordinates
            # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
            xyxy_tensor = detection.xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

            # Get bounding box class ID and name
            classidx = int(detection.cls.item())
            classname = self.labels[classidx]
            print("found " + classname)

            # Get bounding box confidence
            conf = detection.conf.item()

            # Only get confident boxes
            if conf < 0.5:
                continue
            # if (DISPLAY_ENABLED):
            #     drawBox(classidx, frame, xmin, ymin, xmax, ymax, classname)

            if (classname == "person"):
                humans_detected = True
                human_area = (xmax - xmin) * (ymax - ymin)
                if (human_area > biggest_human_area):
                    biggest_human_area = human_area
                    x_mid = (xmax + xmin)/2
                    print("BIGGEST HUMAN AT x: " + str(x_mid))

        if (humans_detected):
            self.state = RobotState.SEEKING_CORRECTION
            self.state_start = self.now
            # Point towards human
            if (x_mid > MIDPOINT and x_mid - MIDPOINT > MARGIN):
                print("rotating counterclockwise")
                self.motors.rotateInPlace(10, False)
            elif (x_mid < MIDPOINT and MIDPOINT - x_mid > MARGIN):
                print("rotating clockwise")
                self.motors.rotateInPlace(10, True)
            else:
                print("moving towards human")
                self.moveToHuman()
        else:
            self.searchMode()
    # Stop spinning the bot and let the next frame search
    def stopSearching(self):
        self.state = RobotState.CHECKING_SEARCH
        self.state_start = self.now
        self.motors.stopRotating()
    def checkRotation(self, cap):
        ret, frame = cap.read()
        results = self.model(frame, verbose=False)
        detections = results[0].boxes
        humans_detected = False
        x_mid = 320
        biggest_human_area = 0

        # Go through each detection and get bbox coords, confidence, and class
        for detection in detections:

            # Get bounding box coordinates
            # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
            xyxy_tensor = detection.xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

            # Get bounding box class ID and name
            classidx = int(detection.cls.item())
            classname = self.labels[classidx]
            print("found " + classname)

            # Get bounding box confidence
            conf = detection.conf.item()

            # Only get confident boxes
            if conf < 0.2:
                continue
            # if (DISPLAY_ENABLED):
            #     drawBox(classidx, frame, xmin, ymin, xmax, ymax, classname)

            if (classname == "person"):
                humans_detected = True
                human_area = (xmax - xmin) * (ymax - ymin)
                if (human_area > biggest_human_area):
                    biggest_human_area = human_area
                    x_mid = (xmax + xmin)/2
                    print("BIGGEST HUMAN AT x: " + str(x_mid))

        if (humans_detected):
            if (abs(x_mid - MIDPOINT) < MARGIN):
                print("moving towards human")
                self.moveToHuman()
        elif (self.state != RobotState.REFINDING_OBJECT):
            # Look harder for humans
            self.state = RobotState.REFINDING_OBJECT
            self.state_start = self.now

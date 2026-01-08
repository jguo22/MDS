import time

from raven import Raven
import cv2
import numpy as np
from ultralytics import YOLO
from enum import Enum

raven_board = Raven()


def drawBox(classidx, frame, xmin, ymin, xmax, ymax, classname):
    color = bbox_colors[classidx % 10]
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

    label = f'{classname}: {int(conf*100)}%'
    labelSize, baseLine = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # Get font size
    # Make sure not to draw label too close to top of window
    label_ymin = max(ymin, labelSize[1] + 10)
    cv2.rectangle(
        frame,
        (xmin,
         label_ymin -
         labelSize[1] -
         10),
        (xmin +
         labelSize[0],
         label_ymin +
         baseLine -
         10),
        color,
        cv2.FILLED)  # Draw white box to put label text in
    cv2.putText(frame, label, (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Draw label text


# Configuration
CAMERA_INDEX = 0
SERVO_CHANNEL = 4
MIN_ANGLE = -90
MAX_ANGLE = 90
MIN_US = 500
MAX_US = 2500
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASS = None
CENTER_DEADZONE = 5
DISPLAY_ENABLED = False

MIDPOINT = 320
MARGIN = 40


class RobotState(Enum):
    SEARCHING = 1
    CHECKING_SEARCH = 2
    SEEKING_CORRECTION = 3
    PICKING_UP = 4
    RETURNING = 5
    DROPPING_OFF = 6
    SEEKING_MOVING = 3


class RavenServoController:
    def __init__(
            self,
            servo_channel=1,
            min_angle=-90,
            max_angle=90,
            min_us=500,
            max_us=2500,
            use_servo=True):
        self.servo_channel = servo_channel
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_us = min_us
        self.max_us = max_us
        self.use_servo = use_servo
        self.current_angle = 0

        print(f"Using Servo: {use_servo}")
        if use_servo:
            self.raven_board = raven_board

            # Set servo channel
            if servo_channel == 1:
                self.channel = Raven.ServoChannel.CH3
            elif servo_channel == 2:
                self.channel = Raven.ServoChannel.CH2
            elif servo_channel == 3:
                self.channel = Raven.ServoChannel.CH3
            elif servo_channel == 4:
                self.channel = Raven.ServoChannel.CH4
            else:
                raise Exception("Invalid Servo Channel")

            # Initialize servo to center position
            self.set_angle(0)

            print(f"Servo initialized on channel {servo_channel}")
            print(f"Angle range: {min_angle}° to {max_angle}°")
            print(f"Pulse width: {min_us}us to {max_us}us")
        else:
            print(f"Simulation mode: Servo channel {servo_channel}")
            print(f"Angle range: {min_angle}° to {max_angle}°")

    def set_angle(self, angle):
        """
        Set servo angle based on target position
        angle: angle in degrees (within min_angle to max_angle range)
        """
        # Clamp angle to min/max range
        angle = max(self.min_angle, min(self.max_angle, angle))

        if self.use_servo:
            self.raven_board.set_servo_position(
                self.channel, angle, min_us=self.min_us, max_us=self.max_us)
        else:
            print(f"[SIMULATION] Servo angle: {angle:.3f}°")

        self.current_angle = angle

    def cleanup(self):
        """Clean up servo controller"""
        if self.use_servo:
            print("Servo stopped")


class RavenMotorControllers:
    def __init__(
            self):

        self.leftChannel = Raven.MotorChannel.CH3
        self.rightChannel = Raven.MotorChannel.CH2
        self.rotating = False

        raven_board.set_motor_encoder(self.leftChannel, 1)
        raven_board.set_motor_mode(self.leftChannel, Raven.MotorMode.DIRECT)
        raven_board.set_motor_encoder(self.rightChannel, 1)
        raven_board.set_motor_mode(self.rightChannel, Raven.MotorMode.DIRECT)

    def setTorque(self, torque):
        raven_board.set_motor_torque_factor(self.leftChannel, torque)
        raven_board.set_motor_torque_factor(self.rightChannel, torque)

    def setSpeed(self, speed, reverse=False):
        raven_board.set_motor_speed_factor(
            self.leftChannel, speed, reverse=reverse)
        raven_board.set_motor_speed_factor(
            self.rightChannel, speed, reverse=not reverse)

    def rotateInPlace(self, speed, clockwise=True):
        self.rotating = True
        raven_board.set_motor_torque_factor(self.leftChannel, 20)
        raven_board.set_motor_torque_factor(self.rightChannel, 20)
        raven_board.set_motor_speed_factor(
            self.leftChannel, speed, reverse=clockwise)
        raven_board.set_motor_speed_factor(
            self.rightChannel, speed, reverse=clockwise)

    def stopRotating(self):
        self.rotating = False
        raven_board.set_motor_torque_factor(self.leftChannel, 0)
        raven_board.set_motor_torque_factor(self.rightChannel, 0)
        raven_board.set_motor_speed_factor(self.leftChannel, 0)
        raven_board.set_motor_speed_factor(self.rightChannel, 0)


class Robot:
    def __init__(self):
        self.state = RobotState.CHECKING_SEARCH
        self.state_start = time.monotonic()
        self.now = time.monotonic()

    def setNowTime(self):
        self.now = time.monotonic()

    # Move towards a human
    def moveToHuman(self):
        self.state = RobotState.SEEKING_MOVING
        self.state_start = self.now
        # Point towards human
        motors.setSpeed(20)
        motors.setTorque(20)
    # Spin around to look for objects

    def searchMode(self):
        self.state = RobotState.SEARCHING
        self.state_start = self.now
        motors.rotateInPlace(40)
    # Check the image. If there's a human, correct. If not, switch to search
    # mode.

    def checkImage(self, humans_found, x_mid):
        if (humans_found):
            self.state = RobotState.SEEKING_CORRECTION
            self.state_start = robot.now
            # Point towards human
            if (x_mid > MIDPOINT and x_mid - MIDPOINT > MARGIN):
                motors.rotateInPlace(5, False)
            elif (x_mid < MIDPOINT and MIDPOINT - x_mid > MARGIN):
                motors.rotateInPlace(5, True)
            else:
                self.moveToHuman
        else:
            self.searchMode()
    # Stop spinning the bot and let the next frame search

    def stopSearching(self):
        self.state = RobotState.CHECKING_SEARCH
        self.state_start = robot.now
        motors.stopRotating()


# Initialize Raven servo controller
servo = RavenServoController(
    servo_channel=SERVO_CHANNEL,
    min_angle=MIN_ANGLE,
    max_angle=MAX_ANGLE,
    min_us=MIN_US,
    max_us=MAX_US,
)

motors = RavenMotorControllers()

robot = Robot()
# Begin inference loop
try:
    while True:
        motors.setSpeed(100)
        motors.setTorque(100)

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cv2.destroyAllWindows()

    raven_board.set_motor_torque_factor(Raven.MotorChannel.CH3, 0)
    raven_board.set_motor_speed_factor(Raven.MotorChannel.CH3, 0)
    raven_board.set_motor_torque_factor(Raven.MotorChannel.CH2, 0)
    raven_board.set_motor_speed_factor(Raven.MotorChannel.CH2, 0)

raven_board.set_motor_encoder(Raven.MotorChannel.CH3, 0) # Set encoder count for motor 1 to zero
print(raven_board.get_motor_encoder(Raven.MotorChannel.CH3)) # Print encoder count = "0"

raven_board.set_motor_mode(Raven.MotorChannel.CH3, Raven.MotorMode.DIRECT) # Set motor mode to DIRECT

# Speed controlled:
raven_board.set_motor_torque_factor(Raven.MotorChannel.CH3, 100) # Let the motor use all the torque to get to speed factor
raven_board.set_motor_speed_factor(Raven.MotorChannel.CH3, 10, reverse=True) # Spin at 10% max speed in reverse

# Torque controlled:
# raven_board.set_motor_speed_factor(Raven.MotorChannel.CH3, 100) # Make motor try to run at max speed forward
# raven_board.set_motor_torque_factor(Raven.MotorChannel.CH3, 100) # Let it use up to 10% available torque
while True:
    pass

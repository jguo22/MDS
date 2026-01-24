import time
from RavenWrapper import RAVEN_WRAPPER, CAMERA_SERVO, ELEVATOR_SERVO, GRIPPER_SERVO


def moveGripperHeight(height):
    pass


def gripClaw():
    RAVEN_WRAPPER.set_servo_position(GRIPPER_SERVO, 90)


def releaseClaw():
    RAVEN_WRAPPER.set_servo_position(GRIPPER_SERVO, -90)


def gripCan():
    moveElevatorDown()
    gripClaw()
    time.sleep(0.5)
    moveElevatorUp()


def moveElevatorUp():
    RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, 90)
    time.sleep(1.2)
    RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, 0)


def moveElevatorDown():
    RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, -90)
    time.sleep(1.2)
    RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, 0)


def rotateCameraDown():
    RAVEN_WRAPPER.set_servo_position(CAMERA_SERVO, -90)


def rotateCameraUp():
    RAVEN_WRAPPER.set_servo_position(CAMERA_SERVO, 30)


def set_camera_angle(self, degree: float) -> bool:
    """Set camera servo angle in degrees"""
    return RAVEN_WRAPPER.set_servo_position(CAMERA_SERVO, degree)

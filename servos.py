import time
from RavenWrapper import RAVEN_WRAPPER, ELEVATOR_SERVO, GRIPPER_SERVO


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

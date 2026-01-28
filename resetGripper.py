import time
from RavenWrapper import ELEVATOR_SERVO, RAVEN_WRAPPER


RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, -90)  # Elevator
sleep_time = 1.3
if (sleep_time < 0):
    sleep_time = 0
time.sleep(sleep_time)
RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, 0)  # Elevator

RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, 90)  # Elevator
sleep_time = 1.4
if (sleep_time < 0):
    sleep_time = 0
time.sleep(sleep_time)
RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, 0)  # Elevator

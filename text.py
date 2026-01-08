from raven import Raven
import math
import time

LEFT_MOTOR = Raven.MotorChannel.CH2
RIGHT_MOTOR = Raven.MotorChannel.CH3

TICK_ROTATION = 64 * 50
WHEEL_D = 90 # TODO: MEASURE ACCIURATELY AND USE MM
BASE_D = 250 # DISTANCE BETEEN CENTER OF WHEELS
# ORIGIN OF ROBOT IS BETWEEN CENTER OF WHEELS
MAX_VELOCITY = 2 * TICK_ROTATION # ticks/s
ACCELERATION = 2 * TICK_ROTATION # ticks/s^2. Reach max v in 1s

BASE_RATIO = WHEEL_D/BASE_D
TURN_CONSTANT = BASE_RATIO * 2 * math.pi/TICK_ROTATION

angle = 0
start_angle = 0
start_left = 0
start_right = 0
total_distance = 0

left_coef = 0
right_coef = 0
last_speed = 0
current_distance = 0

raven = Raven()

for motor in [LEFT_MOTOR, RIGHT_MOTOR]:
    raven.set_motor_encoder(motor, 0)
    raven.set_motor_max_current(motor, 5)
    raven.set_motor_mode(motor, Raven.MotorMode.POSITION)
    raven.set_motor_pid(motor, p_gain = 25, i_gain = 5, d_gain = 0.3)
    raven.set_motor_target(motor, 0)

def startPath(left_coefficient, right_coefficient, distance):
    global total_distance, left_coef, right_coef, start_angle, start_left, start_right
    total_distance = distance
    left_coef = left_coefficient
    right_coef = right_coefficient
    start_angle = angle
    start_left = raven.get_motor_encoder(LEFT_MOTOR)
    start_right = raven.get_motor_encoder(RIGHT_MOTOR)

def updatePath(dt):
    global current_distance, last_speed
    delta_speed = ACCELERATION * dt
    target_speed = 0
    if (total_distance - current_distance <= last_speed ** 2 / ( 2 * ACCELERATION )):
        target_speed = max(last_speed - delta_speed, 0)
    else:
        target_speed = min(last_speed + delta_speed, MAX_VELOCITY)

    current_distance += (last_speed + target_speed)/2*dt
    last_speed = target_speed

    raven.set_motor_target(LEFT_MOTOR, start_left - (current_distance * left_coef))
    raven.set_motor_target(RIGHT_MOTOR, start_right + (current_distance * right_coef))

startPath(1,1, TICK_ROTATION * 5)

while True:
    updatePath(.05)
    time.sleep(.05) # TODO: Measure the dt and set the .05 properly

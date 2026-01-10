from raven import Raven
import math

# Miguel's Navigation movement class
LEFT_MOTOR = Raven.MotorChannel.CH2
RIGHT_MOTOR = Raven.MotorChannel.CH3
TICK_ROTATION = 64 * 50
WHEEL_D = 95 # TODO: MEASURE ACCIURATELY AND USE MM
BASE_D = 209 # DISTANCE BETEEN CENTER OF WHEELS
# ORIGIN OF ROBOT IS BETWEEN CENTER OF WHEELS
ANGLE_PROP = 500

BASE_RATIO = WHEEL_D/BASE_D
TURN_CONSTANT = BASE_RATIO * 2 * math.pi/TICK_ROTATION

class nav:

    def __init__(self):
        self.max_velocity = 10 * TICK_ROTATION # ticks/s
        self.acceleration = 1.5 * TICK_ROTATION # ticks/s^2. Reach max v in 1s

        self.angle = 0
        self.start_angle = 0
        self.start_left = 0
        self.start_right = 0
        self.total_distance = 0

        self.left_coef = 0
        self.right_coef = 0
        self.last_speed = 0
        self.current_distance = 0

        self.raven = Raven()

        for motor in [LEFT_MOTOR, RIGHT_MOTOR]:
            self.raven.set_motor_encoder(motor, 0)
            self.raven.set_motor_max_current(motor, 5)
            self.raven.set_motor_mode(motor, Raven.MotorMode.POSITION)
            self.raven.set_motor_target(motor, 0)

        self.raven.set_motor_pid(RIGHT_MOTOR, p_gain = 25, i_gain = 5, d_gain = 0.13)
        self.raven.set_motor_pid(LEFT_MOTOR, p_gain = 20, i_gain = 5, d_gain = 0.1)

    def startPath(self, left_coefficient, right_coefficient, distance):
        self.total_distance = distance
        self.current_distance = 0
        self.left_coef = left_coefficient
        self.right_coef = right_coefficient
        self.start_angle = self.angle
        self.start_left = self.raven.get_motor_encoder(LEFT_MOTOR)
        self.start_right = self.raven.get_motor_encoder(RIGHT_MOTOR)

    def updatePath(self, dt):
        delta_speed = self.acceleration * dt
        target_speed = 0
        if (self.total_distance - self.current_distance <= self.last_speed ** 2 / ( 2 * self.acceleration )):
            target_speed = max(self.last_speed - delta_speed, 0)
        else:
            target_speed = min(self.last_speed + delta_speed, self.max_velocity)

        self.current_distance += (self.last_speed + target_speed)/2*dt
        self.last_speed = target_speed

        target_angle = (self.right_coef + self.left_coef)/2.0 * TURN_CONSTANT * self.current_distance

        angle_error = (self.angle - self.start_angle) - target_angle

        self.start_left -= angle_error * ANGLE_PROP * dt
        self.start_right -= angle_error * ANGLE_PROP * dt

        self.raven.set_motor_target(LEFT_MOTOR, self.start_left - (self.current_distance * self.left_coef))
        self.raven.set_motor_target(RIGHT_MOTOR, self.start_right + (self.current_distance * self.right_coef))

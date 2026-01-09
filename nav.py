from raven import Raven
import math


class Nav:
    def __init__(self):
        self.LEFT_MOTOR = Raven.MotorChannel.CH2
        self.RIGHT_MOTOR = Raven.MotorChannel.CH3
        self.TICK_ROTATION = 64 * 50
        self.WHEEL_D = 95  # TODO: MEASURE ACCIURATELY AND USE MM
        self.BASE_D = 209  # DISTANCE BETEEN CENTER OF WHEELS
        # ORIGIN OF ROBOT IS BETWEEN CENTER OF WHEELS
        self.MAX_VELOCITY = 2 * self.TICK_ROTATION  # ticks/s
        self.ACCELERATION = 1.5 * self.TICK_ROTATION  # ticks/s^2. Reach max v in 1s

        self.BASE_RATIO = self.WHEEL_D / self.BASE_D
        self.TURN_CONSTANT = self.BASE_RATIO * 2 * math.pi / self.TICK_ROTATION

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

        for motor in [self.LEFT_MOTOR, self.RIGHT_MOTOR]:
            self.raven.set_motor_encoder(motor, 0)
            self.raven.set_motor_max_current(motor, 5)
            self.raven.set_motor_mode(motor, Raven.MotorMode.POSITION)
            self.raven.set_motor_target(motor, 0)

        self.raven.set_motor_pid(
            self.RIGHT_MOTOR,
            p_gain=25,
            i_gain=5,
            d_gain=0.1)
        self.raven.set_motor_pid(
            self.LEFT_MOTOR,
            p_gain=20,
            i_gain=5,
            d_gain=0.1)

    def startPath(self, left_coefficient, right_coefficient, distance):
        print("nav distance " + str(distance))
        self.total_distance = distance
        self.current_distance = 0
        self.left_coef = left_coefficient
        self.right_coef = right_coefficient
        self.start_angle = self.angle
        self.start_left = self.raven.get_motor_encoder(self.LEFT_MOTOR)
        self.start_right = self.raven.get_motor_encoder(self.RIGHT_MOTOR)

    def updatePath(self, dt):
        delta_speed = self.ACCELERATION * dt
        target_speed = 0
        if (self.total_distance - self.current_distance <=
                self.last_speed ** 2 / (2 * self.ACCELERATION)):
            target_speed = max(self.last_speed - delta_speed, 0)
        else:
            target_speed = min(
                self.last_speed + delta_speed,
                self.MAX_VELOCITY)

        self.current_distance += (self.last_speed + target_speed) / 2 * dt
        self.last_speed = target_speed

        self.raven.set_motor_target(
            self.LEFT_MOTOR, self.start_left - (self.current_distance * self.left_coef))
        self.raven.set_motor_target(
            self.RIGHT_MOTOR, self.start_right + (self.current_distance * self.right_coef))

    def start_forward_mm(self, distance_mm):
        distance = distance_mm * self.TICK_ROTATION
        self.startPath(1, 1, distance)

    def start_rotate(self, theta):
        # make into range -pi to pi
        theta = theta % (2 * math.pi)
        if theta > math.pi:
            theta -= 2 * math.pi
        if theta < -math.pi:
            theta += 2 * math.pi

        if theta >= 0:
            self.startPath(
                1, -1, self.TICK_ROTATION * theta)
        else:
            self.startPath(
                -1, 1, self.TICK_ROTATION * -theta)
